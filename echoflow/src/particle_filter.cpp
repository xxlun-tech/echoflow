/** Copyright © 2025 Seaward Science. */

#include "echoflow/particle_filter.hpp"

#include <grid_map_core/iterators/iterators.hpp>

#include <rclcpp/rclcpp.hpp>

namespace echoflow
{

MultiTargetParticleFilter::MultiTargetParticleFilter(size_t num_particles, double initial_max_speed)
: num_particles_(num_particles), initial_max_speed_(initial_max_speed)
{
  particles_.resize(num_particles_);
  rng_.seed(std::random_device{}());
}

bool MultiTargetParticleFilter::initialize(std::shared_ptr<grid_map::GridMap> map_ptr)
{
  if (!map_ptr || !map_ptr->exists("intensity")) {
    RCLCPP_WARN(
      rclcpp::get_logger("MultiTargetParticleFilter"),
      "GridMap missing or lacks 'intensity' layer.");
    return false;
  }

  // Gather valid positions from the map
  std::vector<grid_map::Position> valid_positions = getValidPositionsFromMap(map_ptr);

  if (valid_positions.empty()) {
    RCLCPP_WARN(
      rclcpp::get_logger("MultiTargetParticleFilter"),
      "No valid positions with intensity > 0 found.");
    return false;
  }

  std::uniform_real_distribution<double> uniform_01(
    0.0, 1.0);  // Uniform distribution for random sampling

  // Add new particles at randomly selected valid positions
  for (size_t i = 0; i < num_particles_; ++i) {
    const auto & position = valid_positions[rng_() % valid_positions.size()];
    Target particle;
    particle.x = position.x();
    particle.y = position.y();
    particle.speed = initial_max_speed_ * uniform_01(rng_);
    particle.course = 2.0 * M_PI * uniform_01(rng_);
    particle.yaw_rate = 0.0;  // THIS IS ALWAYS ZERO
    particle.weight = 1.0 / static_cast<double>(num_particles_);
    particle.obs_likelihood = 0.0;
    particle.age = 0.0;                          // Initialize age to zero
    particles_.push_back(particle);
  }
  return true;
}

void MultiTargetParticleFilter::resample(
  std::shared_ptr<grid_map::GridMap> map_ptr, std::shared_ptr<grid_map::GridMap> stats_ptr,
  double dt)
{
  const size_t n_total = num_particles_;
  const size_t n_seed = seed_fraction_ * dt * static_cast<double>(n_total);
  const size_t n_resample = n_total - n_seed;

  std::vector<Target> new_particles;
  new_particles.reserve(n_total);

  // Step 1: Resample n_resample particles
  std::uniform_real_distribution<double> dist_u(0.0, 1.0);
  double step = 1.0 / static_cast<double>(n_resample);
  double initial_offset = dist_u(rng_) * step;
  double cumulative_weight = particles_[0].weight;
  size_t i = 0;

  for (size_t resample_idx = 0; resample_idx < n_resample; ++resample_idx) {
    double uniform_sample_point = initial_offset + resample_idx * step;
    while (uniform_sample_point > cumulative_weight && i < particles_.size() - 1) {
      ++i;
      cumulative_weight += particles_[i].weight;
    }
    Target particle = particles_[i];
    addResampleNoise(particle);
    particle.weight = 1.0 / n_total;
    particle.age = particles_[i].age;  // Preserve age from original particle
    new_particles.push_back(particle);
  }

  // Step 2: Inject n_seed randomly initialized particles
  std::vector<grid_map::Position> valid_positions = getValidPositionsFromMap(map_ptr);

  if (stats_ptr) {
    seedWeighted(valid_positions, n_seed, stats_ptr, new_particles);
  } else {
    seedUniform(valid_positions, n_seed, new_particles);
  }

  particles_ = std::move(new_particles);
}

void MultiTargetParticleFilter::predict(double dt)
{
  for (auto & particle : particles_) {
    double velocity = particle.speed + noise_speed_(rng_);
    double yaw = particle.course + noise_yaw_(rng_);
    double omega = particle.yaw_rate + noise_yaw_rate_(rng_);

    if (std::abs(omega) > 1e-3) {
      double radius = velocity / omega;
      particle.x += radius * (std::sin(yaw + omega * dt) - std::sin(yaw)) + noise_position_(rng_);
      particle.y += radius * (-std::cos(yaw + omega * dt) + std::cos(yaw)) + noise_position_(rng_);
    } else {
      particle.x += velocity * std::cos(yaw) * dt + noise_position_(rng_);
      particle.y += velocity * std::sin(yaw) * dt + noise_position_(rng_);
    }

    particle.course += omega * dt;
    particle.course = std::fmod(particle.course + 2 * M_PI, 2 * M_PI);
    particle.age += dt;  // Increment particle age
  }
}

void MultiTargetParticleFilter::updateWeights(
  std::shared_ptr<grid_map::GridMap> map_ptr, std::shared_ptr<grid_map::GridMap> stats_ptr,
  double dt)
{
  if (!map_ptr || !map_ptr->exists("edt")) {
    RCLCPP_WARN(
      rclcpp::get_logger("MultiTargetParticleFilter"), "GridMap does not contain 'edt' layer.");
    return;
  }

  double total_weight = 0.0;
  const double sigma = observation_sigma_;
  double half_life_decay = std::exp(-std::log(2.0) * dt / weight_decay_half_life_);

  auto cell_size = stats_ptr->getResolution();
  auto cell_area = cell_size * cell_size;
  double steepness = 5.0 / density_feedback_factor_;  // controls the steepness of the curve
  const double inv2sig2 = 1.0 / (2.0 * sigma * sigma);

  for (auto & particle : particles_) {
    grid_map::Position position(particle.x, particle.y);
    double obs_likelihood = 1e-6;  // small baseline to prevent zero weights

    if (map_ptr->isInside(position)) {
      double distance = map_ptr->atPosition("edt", position);
      obs_likelihood = std::exp(-distance * distance * inv2sig2);
    }

    if (particle.obs_likelihood < obs_likelihood) {
      particle.obs_likelihood = obs_likelihood;
    } else {
      particle.obs_likelihood *= half_life_decay;
    }
    particle.weight = std::max(particle.obs_likelihood, 1e-8);

    // Penalize overcrowded areas using a logistic decay
    if (stats_ptr && stats_ptr->isInside(position)) {
      double density =
        stats_ptr->atPosition(
          "particles_per_cell", position, grid_map::InterpolationMethods::INTER_NEAREST) /
        cell_area;
      if (!std::isnan(density) && density > 0.0) {
        // Logistic penalty: penalty ≈ 1.0 when density << threshold, ≈ 0.0 when density >>
        // threshold
        double x = density - density_feedback_factor_;
        double density_penalty = 1.0 / (1.0 + std::exp(steepness * x));
        particle.weight *= density_penalty;
      }
    }

    total_weight += particle.weight;
  }

  // Normalize weights
  if (total_weight > 0.0) {
    for (auto & particle : particles_) {
      particle.weight /= total_weight;
    }
  }
}

std::vector<grid_map::Position> MultiTargetParticleFilter::getValidPositionsFromMap(
  const std::shared_ptr<grid_map::GridMap> & map_ptr)
{
  std::vector<grid_map::Position> valid_positions;

  for (grid_map::GridMapIterator it(*map_ptr); !it.isPastEnd(); ++it) {
    const auto & index = *it;
    if (!map_ptr->isValid(index, "targets")) continue;

    double val = map_ptr->at("targets", index);
    if (std::isnan(val) || val <= 0.0) continue;

    grid_map::Position position;
    if (map_ptr->getPosition(index, position)) {
      valid_positions.push_back(position);
    }
  }

  return valid_positions;
}

void MultiTargetParticleFilter::addResampleNoise(Target & particle)
{
  particle.x += noise_position_(rng_);
  particle.y += noise_position_(rng_);

  particle.course += noise_yaw_(rng_);
  // Wrap course angle to [0, 2π)
  particle.course = std::fmod(particle.course, 2.0 * M_PI);
  if (particle.course < 0.0) particle.course += 2.0 * M_PI;

  particle.speed += noise_speed_(rng_);
  // Ensure speed is non-negative
  particle.speed = std::max(0.0, particle.speed);

  // particle.yaw_rate += noise_yaw_rate_(rng_); // TODO
}

void MultiTargetParticleFilter::seedUniform(
  const std::vector<grid_map::Position> & valid_positions, size_t n_seed,
  std::vector<Target> & output_particles)
{
  std::uniform_real_distribution<double> uniform_01(0.0, 1.0);
  for (size_t m = 0; m < n_seed && !valid_positions.empty(); ++m) {
    const auto & position = valid_positions[rng_() % valid_positions.size()];
    Target particle;
    particle.x = position.x();
    particle.y = position.y();
    particle.speed = initial_max_speed_ * uniform_01(rng_);
    particle.course = 2.0 * M_PI * uniform_01(rng_);
    particle.yaw_rate = 0.0;
    particle.weight = 1.0 / static_cast<double>(num_particles_);
    particle.obs_likelihood = 0.0;
    particle.age = 0.0;  // Seed age at zero
    output_particles.push_back(particle);
  }
}

void MultiTargetParticleFilter::seedWeighted(
  const std::vector<grid_map::Position> & valid_positions, size_t n_seed,
  const std::shared_ptr<grid_map::GridMap> & stats_ptr, std::vector<Target> & output_particles)
{
  // build weight vector using particles per cell
  std::vector<double> weights;
  weights.reserve(valid_positions.size());
  constexpr double eps = 1e-3;
  for (auto & position : valid_positions) {
    double density = 0.0;
    if (stats_ptr->isInside(position)) {
      density = stats_ptr->atPosition(
        "particles_per_cell", position);  // lookup local density at valid position
      if (std::isnan(density)) {
        density = 0.0;
      }
    }
    weights.push_back(
      1.0 / (density + eps));  // inverse density as weight, avoid divide by zero w/ eps
  }

  // discrete distribution based on weights
  // cumulative distribution so higher weights more likely to be chosen
  std::discrete_distribution<size_t> sampler(weights.begin(), weights.end());
  std::uniform_real_distribution<double> uniform_01(0.0, 1.0);

  for (size_t m = 0; m < n_seed && !valid_positions.empty(); ++m) {
    size_t idx = sampler(rng_);
    const auto & position = valid_positions[idx];
    Target particle;
    particle.x = position.x();
    particle.y = position.y();
    particle.speed = initial_max_speed_ * uniform_01(rng_);
    particle.course = 2.0 * M_PI * uniform_01(rng_);
    particle.yaw_rate = 0.0;
    particle.weight = 1.0 / static_cast<double>(num_particles_);
    particle.obs_likelihood = 0.0;
    particle.age = 0.0;  // Seed age at zero
    output_particles.push_back(particle);
  }
}

void MultiTargetParticleFilter::updateNoiseDistributions()
{
  noise_position_ = std::normal_distribution<double>(0.0, noise_std_position_);
  noise_yaw_ = std::normal_distribution<double>(0.0, noise_std_yaw_);
  noise_yaw_rate_ = std::normal_distribution<double>(0.0, noise_std_yaw_rate_);
  noise_speed_ = std::normal_distribution<double>(0.0, noise_std_speed_);
}

const std::vector<Target> & MultiTargetParticleFilter::getParticles()
{
  return particles_;
}

}

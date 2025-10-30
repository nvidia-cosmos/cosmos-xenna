// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A copy of pipelines::private::scheduling::autoscaling_algorithms::tests::test_large_number_of_nodes
//! This was copied because I couldn't figure out how to get this command working:
//! CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --release --unit-test cosmos-xenna -- pipelines::private::scheduling::autoscaling_algorithms::tests::test_large_number_of_nodes
//! TODO: get this working. I think it has something to do with how we declare the lib as _cosmos_xenna instead of cosmos_xenna

use _cosmos_xenna::pipelines::private::scheduling::autoscaling_algorithms::{
    Estimate, Estimates, WorkerIdFactory, run_fragmentation_autoscaler,
};
use _cosmos_xenna::pipelines::private::scheduling::data_structures as ds;
use _cosmos_xenna::pipelines::private::scheduling::resources as rds;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn make_cluster(
    num_nodes: usize,
    cpus_per_node: usize,
    gpus_per_node: usize,
    heterogeneous: bool,
) -> rds::ClusterResources {
    let mut nodes: std::collections::HashMap<String, rds::NodeResources> =
        std::collections::HashMap::new();
    for i in 0..num_nodes {
        let mut cpus = cpus_per_node as f32;
        let mut gpus = gpus_per_node;
        if heterogeneous && (i % 2 == 0) {
            cpus = (cpus_per_node as f32) / 2.0;
            gpus = gpus_per_node / 2;
        }
        let mut gpu_vec: Vec<rds::GpuResources> = Vec::new();
        for _ in 0..gpus {
            gpu_vec.push(rds::GpuResources {
                index: i as u8,
                uuid_: uuid::Uuid::new_v4(),
                used_fraction: rds::FixedUtil::ZERO,
            });
        }
        nodes.insert(
            format!("node{}", i),
            rds::NodeResources {
                used_cpus: rds::FixedUtil::ZERO,
                total_cpus: rds::FixedUtil::from_num(cpus),
                gpus: gpu_vec,
                name: None,
            },
        );
    }
    rds::ClusterResources { nodes }
}

fn make_default_state_for_stages(problem: &ds::Problem) -> ds::ProblemState {
    ds::ProblemState {
        stages: problem
            .stages
            .iter()
            .map(|s| ds::ProblemStageState {
                stage_name: s.name.clone(),
                worker_groups: Vec::new(),
                slots_per_worker: 2,
                is_finished: false,
            })
            .collect(),
    }
}

fn estimates_from_speeds(speeds: &[Option<f64>]) -> Estimates {
    Estimates {
        stages: speeds
            .iter()
            .map(|s| Estimate {
                batches_per_second_per_worker: *s,
                num_returns_per_batch: Some(1.0),
            })
            .collect(),
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let num_nodes = 1000;
    // Create a cluster with plenty of resources
    let cluster = make_cluster(
        num_nodes, // 100 nodes
        240,       // 240 CPUs per node
        8,         // 8 GPUs per node
        false,
    );

    let mut stages = Vec::new();
    let mut speeds = Vec::new();
    let mut cur_stage_idx = 0;

    // Fixed CPU stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: Some((num_nodes * 4) as usize),
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Flexible CPU stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Larger flexible CPU stage
    // Flexible CPU stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Small GPU stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.25,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Another large cpu stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 6.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Another large cpu stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Another small cpu stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Another small gpu stage stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.25,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Another medium cpu stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 4.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Medium GPU stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 1.0,
            cpus: 1.0,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: None,
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));
    cur_stage_idx += 1;

    // Fixed CPU stage
    stages.push(ds::ProblemStage {
        name: format!("stage_{}", cur_stage_idx),
        stage_batch_size: 1,
        worker_shape: rds::Resources {
            gpus: 0.0,
            cpus: 0.25,
            is_spmd: false,
        }
        .to_shape(&cluster)
        .unwrap(),
        requested_num_workers: Some((num_nodes * 8) as usize),
        over_provision_factor: None,
    });
    speeds.push(Some(1.0));

    let problem = ds::Problem {
        cluster_resources: cluster,
        stages,
    };

    let state = make_default_state_for_stages(&problem);
    let estimates = estimates_from_speeds(&speeds);

    log::info!("Running autoscaler with 1.5x over-provision factor");
    let mut worker_id_factory = WorkerIdFactory::new();
    let solution1 =
        run_fragmentation_autoscaler(&problem, &state, &estimates, 1.5, &mut worker_id_factory);
    log::info!("Finished first run");

    // Multiply the speeds by some random (repeatable) factor.
    let mut rng = StdRng::seed_from_u64(12345);
    speeds = speeds
        .into_iter()
        .map(|opt| opt.map(|v| v * rng.random_range(0.1..10.0)))
        .collect();
    log::info!("Speeds: {:?}", speeds);

    let estimates = estimates_from_speeds(&speeds);
    let new_state = ds::ProblemState {
        stages: problem
            .stages
            .iter()
            .zip(solution1.stages.iter())
            .map(|(p, s)| ds::ProblemStageState {
                stage_name: p.name.clone(),
                worker_groups: s.new_workers.clone(),
                slots_per_worker: s.slots_per_worker,
                is_finished: false,
            })
            .collect(),
    };

    let _solution2 = run_fragmentation_autoscaler(
        &problem,
        &new_state,
        &estimates,
        1.5,
        &mut worker_id_factory,
    );

    // Print a small confirmation so we know it ran
    log::info!(
        "Ran large autoscaling problem: stages={} nodes={} gpus={} cpus={}",
        problem.stages.len(),
        problem.cluster_resources.num_nodes(),
        problem.cluster_resources.num_total_gpus(),
        problem.cluster_resources.num_total_cpus()
    );
}

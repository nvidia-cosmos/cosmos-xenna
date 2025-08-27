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

use env_logger::Env;
use log::debug;
use pyo3::prelude::*;

#[pyfunction]
pub fn setup_logging() {
    if let Err(e) =
        env_logger::Builder::from_env(Env::default().default_filter_or("warn")).try_init()
    {
        debug!("Failed to initialize logger: {}", e);
    }
}

#[pymodule]
pub fn logging(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(setup_logging, m)?)?;
    Ok(())
}
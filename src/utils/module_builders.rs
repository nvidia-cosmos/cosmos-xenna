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

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyNone};
use pyo3::{PyClass, ffi};
use std::ffi::CString;

/// A builder pattern implementation for creating Python modules that can be properly
/// imported in Python code.
///
/// This builder handles the necessary setup for modules to be importable, including:
/// - Setting up module hierarchy
/// - Configuring `__path__`, `__file__`, and other required attributes
/// - Adding submodules and classes with proper `__module__` attributes
///
/// # Example
/// ```rust
/// use _cosmos_xenna::utils::module_builders::ImportablePyModuleBuilder;
/// use pyo3::prelude::*;
/// use pyo3::wrap_pyfunction;
///
/// fn example_usage() -> PyResult<()> {
///     Python::with_gil(|py| {
///         let m = PyModule::new(py, "my_module")?;
///         let my_module = ImportablePyModuleBuilder::new(py, "my_package.my_module")?
///             .add_function(wrap_pyfunction!(my_function, m)?)?
///             .finish();
///         Ok(())
///     })
/// }
///
/// #[pyfunction]
/// fn my_function() -> &'static str {
///     "Hello, world!"
/// }
/// ```
pub struct ImportablePyModuleBuilder<'py> {
    inner: Bound<'py, PyModule>,
    name: String,
}

impl<'py> ImportablePyModuleBuilder<'py> {
    /// Creates a new module builder with the given fully qualified name.
    ///
    /// # Arguments
    /// * `py` - The Python interpreter
    /// * `name` - The fully qualified module name (e.g., "package.subpackage.module")
    ///
    /// # Safety
    /// This function uses PyImport_AddModule which is safe when called with a valid Python
    /// interpreter and a properly formatted module name.
    pub fn new(py: Python<'py>, name: &str) -> PyResult<Self> {
        // Create a CString that lives through the function call
        let c_name = CString::new(name)?;

        // Create the module
        let module = unsafe {
            let ptr = ffi::PyImport_AddModule(c_name.as_ptr());
            if ptr.is_null() {
                return Err(PyErr::fetch(py));
            }

            // Create a bound reference and downcast
            let bound = Bound::from_borrowed_ptr(py, ptr);
            bound.downcast_into::<PyModule>()?
        };

        // Initialize basic module attributes
        module.setattr("__file__", PyNone::get(py))?;

        // Set __package__ attribute (parent package name)
        let package_name = name
            .rsplit_once('.')
            .map(|(prefix, _)| prefix)
            .unwrap_or("");
        module.setattr("__package__", package_name)?;

        // Initialize __path__ as an empty list for package-like behavior
        if !module.hasattr("__path__")? {
            module.setattr("__path__", PyList::empty(py))?;
        }

        // Create an empty __dict__ if not present
        if !module.hasattr("__dict__")? {
            module.setattr("__dict__", PyDict::new(py))?;
        }

        Ok(Self {
            inner: module,
            name: name.to_string(),
        })
    }

    /// Adds a submodule to this module.
    ///
    /// # Arguments
    /// * `module` - The submodule to add
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_submodule(self, module: &Bound<'_, PyModule>) -> PyResult<Self> {
        // Extract the fully qualified name from the module
        let fully_qualified_name: String = module.name()?.extract()?;

        // Extract the simple name (last part after dot)
        let name = match fully_qualified_name.rsplit_once('.') {
            Some((_, name)) => name,
            None => &fully_qualified_name,
        };

        // Add the module with its simple name
        self.inner.add(name, module)?;

        // Ensure __path__ is set for package-like behavior
        if !self.inner.hasattr("__path__")? {
            self.inner
                .setattr("__path__", PyList::empty(self.inner.py()))?;
        }

        Ok(self)
    }

    /// Adds a Python class to this module and sets its __module__ attribute correctly.
    ///
    /// # Type Parameters
    /// * `T` - The PyClass to add
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_class<T: PyClass>(self) -> PyResult<Self> {
        // Add the class to the module
        self.inner.add_class::<T>()?;

        // Update the __module__ attribute to the correct module name
        let py = self.inner.py();
        let type_object = T::lazy_type_object().get_or_init(py);

        // Only override the __module__ if it's set to "builtins" (default)
        let current_module = type_object.getattr("__module__")?.extract::<String>()?;
        if current_module == "builtins" {
            type_object.setattr("__module__", &self.name)?;
        }

        Ok(self)
    }

    /// Adds a function to the module.
    ///
    /// # Arguments
    /// * `name` - The name for the function
    /// * `function` - The Python function to add
    ///
    /// # Returns
    /// Self for method chaining
    pub fn add_function(
        self,
        wrapped_function: Bound<'_, pyo3::types::PyCFunction>,
    ) -> PyResult<Self> {
        let name = wrapped_function.getattr("__name__")?.extract::<String>()?;
        self.inner.add(&name, wrapped_function)?;
        Ok(self)
    }

    /// Creates a builder from an existing PyModule.
    ///
    /// # Arguments
    /// * `module` - The existing module to wrap
    ///
    /// # Returns
    /// A new builder that wraps the provided module
    pub fn from(module: Bound<'py, PyModule>) -> PyResult<Self> {
        let name = module.name()?.extract()?;
        Ok(Self {
            inner: module,
            name,
        })
    }

    /// Completes the building process and returns the constructed module.
    /// Use this when you're done with the builder and want to get the final module.
    ///
    /// # Returns
    /// The built PyModule
    pub fn finish(self) -> Bound<'py, PyModule> {
        self.inner
    }

    /// Returns a reference to the module being built.
    /// Use this when you need temporary access but want to continue building.
    ///
    /// # Returns
    /// A reference to the module
    pub fn as_module(&self) -> &Bound<'py, PyModule> {
        &self.inner
    }
}

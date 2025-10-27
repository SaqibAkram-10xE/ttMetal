// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "example_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/examples/example/example.hpp"

#include "ttnn/types.hpp"


namespace ttnn::operations::examples {

void bind_example_operation(ttnn::py::module& module) {
    using namespace ttnn; 


    bind_registered_operation(
        module,
        ttnn::example,
        R"doc(example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

        // Add pybind overloads for the C++ APIs that should be exposed to python
        // There should be no logic here, just a call to `self` with the correct arguments
        // This specific function can be called from python as `ttnn.prim.example(input_tensor)` or
        // `ttnn.prim.example(input_tensor)`
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::example)& self,
               const ttnn::Tensor& RowIdx_tensor,
               const ttnn::Tensor& CodeBook_tensor,
               const ttnn::Tensor& ColIdx_tensor) -> ttnn::Tensor {
                return self(RowIdx_tensor, CodeBook_tensor, ColIdx_tensor);
            },
            ttnn::py::arg("RowIdx_tensor"),
            ttnn::py::arg("CodeBook_tensor"),
            ttnn::py::arg("ColIdx_tensor")});


    // bind_registered_operation(
    //     module,
    //     ttnn::composite_example,
    //     R"doc(composite_example(input_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

    //     // Add pybind overloads for the C++ APIs that should be exposed to python
    //     // There should be no logic here, just a call to `self` with the correct arguments
    //     ttnn::pybind_overload_t{
    //         [](const decltype(ttnn::composite_example)& self,
    //            const ttnn::Tensor& RowIdx_tensor,
    //            const ttnn::Tensor& CodeBook_tensor,
    //            const ttnn::Tensor& ColIdx_tensor) -> ttnn::Tensor {
    //             return self(RowIdx_tensor, CodeBook_tensor, ColIdx_tensor);
    //         },
    //         ttnn::py::arg("RowIdx_tensor"),
    //         ttnn::py::arg("CodeBook_tensor"),
    //         ttnn::py::arg("ColIdx_tensor")});

}

}  // namespace ttnn::operations::examples

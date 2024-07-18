# mypy: allow-untyped-defs
from typing import Callable, List, Optional

from .. import ir

from ..select_algorithm import DataProcessorTemplateWrapper
from .cpp_gemm_template import CppPackedGemmTemplate, GEMM_TEMPLATE, MICROKERNEL_DEF

from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import GemmBlocking

GEMM_SINGLE_THREAD_MM_STUB = r"""
void single_thread_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input2_t']}}* W,
    {{micro_gemm.get_common_options()['input_t']}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {%- endif %}
)
"""

GEMM_THREADED_MM_STUB = r"""
void threaded_mm(
    const {{micro_gemm.get_common_options()['input_t']}}* X,
    const {{micro_gemm.get_common_options()['input2_t']}}* W,
    {{micro_gemm.get_common_options()['input_t']}}* Y
    {%- if is_dynamic_M %},
    const int64_t {{kernel.size(GemmOut, -2, unwrapped=True)}}
    {%- endif %}
)
"""

BMM_WRAPPER = r"""
extern "C"
{{kernel.def_kernel(inputs={"BX": BX, "BW": BW}, outputs={"BY": BY}, aliases=buffer_aliases)}}
{
    const int64_t B = {{kernel.size(BY, -3, unwrapped=True)}};
    {%- if num_threads > 1 %}
    constexpr int64_t num_threads = {{num_threads}};
    int64_t B_single_thread_block = (B / num_threads) * num_threads;

    #pragma omp parallel for num_threads({{num_threads}})
    {%- else %}
    int64_t B_single_thread_block = B;
    {%- endif %}
    for (int64_t b_start = 0; b_start < B_single_thread_block; ++b_start) {
        single_thread_mm(
            &{{kernel.index(BX, ["b_start", 0, 0])}},
            &{{kernel.index(BW, ["b_start", 0, 0])}},
            &{{kernel.index(BY, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {%- endif %}
        );
    }
    for (int64_t b_start = B_single_thread_block; b_start < B; ++b_start) {
        threaded_mm(
            &{{kernel.index(BX, ["b_start", 0, 0])}},
            &{{kernel.index(BW, ["b_start", 0, 0])}},
            &{{kernel.index(BY, ["b_start", 0, 0])}}
            {%- if is_dynamic_M %},
            {{kernel.size(GemmOut, -2)}}
            {%- endif %}
        );
    }
}
"""


class CppBmmTemplate(CppPackedGemmTemplate):
    def __init__(
        self,
        input_nodes,
        layout: ir.Layout,
        num_threads: int,
        register_blocking: GemmBlocking,
        beta=1,
        alpha=1,
        has_bias=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
        name="bmm",
    ):
        super().__init__(
            input_nodes,
            layout,
            num_threads,
            register_blocking,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            epilogue_creator=epilogue_creator,
            name=name,
        )
        self.should_pack_weights = False

    @staticmethod
    def add_choices(
        choices,
        layout,
        input_nodes,
        beta=1,
        alpha=1,
        has_bias=False,
        trans_w=False,
        input_indices=None,
        should_pack_weights=False,
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,
    ):
        options = super(CppBmmTemplate, CppBmmTemplate)._get_params_for_choices(
            layout=layout,
            input_nodes=input_nodes,
            beta=beta,
            alpha=alpha,
            has_bias=has_bias,
            trans_w=trans_w,
            input_indices=input_indices,
            should_pack_weights=should_pack_weights,
            epilogue_creator=epilogue_creator,
        )
        template = DataProcessorTemplateWrapper(CppBmmTemplate, **options)
        template.maybe_append_choice(choices)
        return template

    def _get_default_reindexers(self, epilogue_nodes):
        def reindexer(args):
            if len(epilogue_nodes) == 0:
                return args
            return [0] + args

        return [reindexer]

    def get_options(self, kernel, template_buffer_node, epilogue_nodes, **kwargs):
        options = super().get_options(
            kernel, template_buffer_node, epilogue_nodes, **kwargs
        )
        BX, BW, BY = options["X"], options["W"], options["Y"]
        options["BX"], options["BW"], options["BY"] = BX, BW, BY
        for kword in ["X", "W", "Y", "GemmOut", "Y_2d"]:
            options[kword] = kernel.select(options[kword], 0, 0)
        return options

    def render(  # type: ignore[override]
        self,
        kernel: CppTemplateKernel,
        template_buffer_node: Optional[ir.CppTemplateBuffer] = None,
        epilogue_nodes: Optional[List[ir.IRNode]] = None,
        **kwargs,
    ) -> str:
        options = self.get_options(
            kernel, template_buffer_node, epilogue_nodes, **kwargs
        )
        BX, BW, BY = options["BX"], options["BW"], options["BY"]
        X, W, Y = options["X"], options["W"], options["Y"]
        buffer_aliases = options["buffer_aliases"]

        kernel.set_args(
            inputs={"X": X, "W": W}, outputs={"Y": Y}, aliases=buffer_aliases
        )
        result = self._template_from_string(MICROKERNEL_DEF).render(**options)
        result += self._template_from_string(GEMM_THREADED_MM_STUB + GEMM_TEMPLATE).render(
            **options
        )
        result += self._template_from_string(GEMM_SINGLE_THREAD_MM_STUB + GEMM_TEMPLATE).render(
            **{**options, "num_threads": 1}
        )
        kernel.set_args(
            inputs={"BX": BX, "BW": BW}, outputs={"BY": BY}, aliases=buffer_aliases
        )
        result += self._template_from_string(BMM_WRAPPER).render(**options)
        return result

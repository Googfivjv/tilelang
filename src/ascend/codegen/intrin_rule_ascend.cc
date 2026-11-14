/*!
 * \file intrin_rule_ascend.cc
 * \brief Ascend intrinsic rules for math functions.
 *
 * The default FloatSuffix lowering uses exact dtype matching
 * (t == DataType::Float(32)) which fails for vectorized types.
 * AscendMath uses t.is_float() like CUDAMath to support any lane count.
 *
 * Registered with "ascend.FLowerIntrinsic" for the split Ascend backend.
 */
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;

struct AscendMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16:
        return "h" + name;
      default:
        return "";
      }
    }
    return "";
  }
};

PrimExpr DispatchAscendMath(const PrimExpr &expr) {
  const auto *call = expr.as<tirx::CallNode>();
  TVM_FFI_ICHECK(call != nullptr);
  if (!call->dtype.is_bfloat16()) {
    return DispatchPureExtern<AscendMath>(expr);
  }

  const DataType result_dtype = call->dtype;
  const DataType compute_dtype = DataType::Float(32, result_dtype.lanes());
  ffi::Array<PrimExpr> args;
  for (const PrimExpr &arg : call->args) {
    args.push_back(
        arg.dtype().is_bfloat16()
            ? tirx::Cast(DataType::Float(32, arg.dtype().lanes()), arg)
            : arg);
  }
  PrimExpr widened =
      tirx::Call(compute_dtype, call->op, args, call->annotations);
  return tirx::Cast(result_dtype, DispatchPureExtern<AscendMath>(widened));
}

#define REGISTER_ASCEND_INTRIN(OpName)                                         \
  TVM_REGISTER_OP("tirx." OpName)                                              \
      .set_attr<FLowerIntrinsic>("ascend.FLowerIntrinsic", DispatchAscendMath)

REGISTER_ASCEND_INTRIN("sqrt");
REGISTER_ASCEND_INTRIN("rsqrt");
REGISTER_ASCEND_INTRIN("exp");
REGISTER_ASCEND_INTRIN("exp2");
REGISTER_ASCEND_INTRIN("exp10");
REGISTER_ASCEND_INTRIN("log");
REGISTER_ASCEND_INTRIN("log2");
REGISTER_ASCEND_INTRIN("log10");
REGISTER_ASCEND_INTRIN("log1p");
REGISTER_ASCEND_INTRIN("tanh");
REGISTER_ASCEND_INTRIN("tan");
REGISTER_ASCEND_INTRIN("sin");
REGISTER_ASCEND_INTRIN("cos");
REGISTER_ASCEND_INTRIN("atan");
REGISTER_ASCEND_INTRIN("atan2");
REGISTER_ASCEND_INTRIN("erf");
REGISTER_ASCEND_INTRIN("floor");
REGISTER_ASCEND_INTRIN("ceil");
REGISTER_ASCEND_INTRIN("fabs");
REGISTER_ASCEND_INTRIN("round");
REGISTER_ASCEND_INTRIN("trunc");

#undef REGISTER_ASCEND_INTRIN

} // namespace intrin
} // namespace codegen
} // namespace tvm

// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

////////////////////////////////////////
// kernel list macros
////////////////////////////////////////

#undef TOTAL_KPARAM_LIST

////////////////////////////////////////
// customized macros
////////////////////////////////////////

#undef TILE_N_PER_CTA
#undef TILE_M_PER_CTA

#undef TILE_K_PER_CTA
#undef TILE_K_PER_STEP

#undef TILE_N_PER_WARP
#undef TILE_M_PER_WARP

#undef KERNEL_NAME

////////////////////////////////////////
// align functions
////////////////////////////////////////

#undef Align
#undef DivUp

#undef Min
#undef Max

////////////////////////////////////////
// boundary check
////////////////////////////////////////

#undef WidthInRange
#undef HeightInRange
#undef BatchInRange

////////////////////////////////////////
// constant cta size macros
////////////////////////////////////////

#undef _4CHAR_TO_INT_
#undef _4INT_TO_INT4_
#undef _2INT_TO_INT2_

#undef _2HALF_TO_INT_
#undef _2INT2_TO_INT4_

#undef _C1_
#undef _C2_
#undef _C4_
#undef _C8_
#undef _C16_
#undef _C32_

#undef _1INT_
#undef _2INT_
#undef _4INT_
#undef _8INT_

#undef _1INT4_
#undef _2INT4_
#undef _4INT4_
#undef _8INT4_

#undef _1INT8_
#undef _2INT8_
#undef _4INT8_
#undef _8INT8_

#undef _1HALF_
#undef _2HALF_
#undef _4HALF_
#undef _8HALF_

#undef _1HALF2_
#undef _2HALF2_
#undef _4HALF2_
#undef _8HALF2_

#undef _1MMA_
#undef _2MMA_
#undef _4MMA_
#undef _8MMA_

#undef _HALF_ZERO_
#undef _ZERO_

#undef _FLOAT_ZERO_
#undef _FLOAT_ONE_

#undef _1INT_X1_
#undef _1INT_X2_
#undef _1INT_X4_

#undef _2INT_X1_
#undef _2INT_X2_
#undef _2INT_X4_

#undef _4INT_X1_
#undef _4INT_X2_
#undef _4INT_X4_

#undef _INT_TO_BYTE_
#undef _INT_TO_2HALF_
#undef _INT2_TO_2HALF2_
#undef _INT2_TO_2INT_
#undef _INT2_TO_4HALF_

#undef _INT4_TO_INT4_
#undef _INT4_TO_2INT2_
#undef _INT4_TO_4INT_
#undef _INT4_TO_4HALF2_
#undef _INT4_TO_8HALF_

////////////////////////////////////////
// mma size macros
////////////////////////////////////////

#undef TILE_M_PER_MMA
#undef TILE_K_PER_MMA
#undef TILE_N_PER_MMA
#undef TILE_M_PER_MMA_HALF
#undef TILE_M_PER_MMA_QTR

#undef MMA_SIZE_Y_IN_THD
#undef MMA_SIZE_Y_IN_THD

#undef MMA_SIZE_X_IN_BITS
#undef CTA_SIZE_X_IN_BITS

#undef BLK_M_PER_MMA
#undef BLK_N_PER_MMA

////////////////////////////////////////
// thread / warp / cta size macros
////////////////////////////////////////

#undef WARP_SIZE_IN_THD
#undef WARP_SIZE_IN_BITS

#undef WARP_SIZE_X_IN_THD
#undef WARP_SIZE_Y_IN_THD

#undef CTA_SIZE_X_IN_WARP
#undef CTA_SIZE_Y_IN_WARP

#undef CTA_SIZE_IN_WARP
#undef CTA_SIZE_IN_THD

#undef WARP_SIZE_IN_THD_HALF
#undef WARP_SIZE_IN_THD_QTR

#undef NUM_M_STEPS
#undef NUM_N_STEPS

////////////////////////////////////////
// tiling size macros
////////////////////////////////////////

#undef TILE_M_PER_STEP
#undef TILE_N_PER_STEP

#undef TILE_M_PER_THD
#undef TILE_N_PER_THD

/////////////////////
// tile m

#undef TILE_M_V1_PER_CTA
#undef TILE_M_V2_PER_CTA
#undef TILE_M_V4_PER_CTA
#undef TILE_M_V8_PER_CTA

#undef TILE_M_V1_PER_WARP
#undef TILE_M_V2_PER_WARP
#undef TILE_M_V4_PER_WARP
#undef TILE_M_V8_PER_WARP

#undef TILE_M_V1_PER_THD
#undef TILE_M_V2_PER_THD
#undef TILE_M_V4_PER_THD
#undef TILE_M_V8_PER_THD

#undef TILE_M_V1_PER_MMA
#undef TILE_M_V2_PER_MMA
#undef TILE_M_V4_PER_MMA
#undef TILE_M_V8_PER_MMA
#undef TILE_M_V1_PER_MMA_HALF

/////////////////////
// tile k

#undef TILE_K_V1_PER_CTA
#undef TILE_K_V2_PER_CTA
#undef TILE_K_V4_PER_CTA
#undef TILE_K_V8_PER_CTA
#undef TILE_K_V16_PER_CTA

#undef TILE_K_V1_PER_STEP
#undef TILE_K_V2_PER_STEP
#undef TILE_K_V4_PER_STEP
#undef TILE_K_V8_PER_STEP
#undef TILE_K_V16_PER_STEP

#undef TILE_K_V1_PER_MMA
#undef TILE_K_V2_PER_MMA
#undef TILE_K_V4_PER_MMA
#undef TILE_K_V8_PER_MMA
#undef TILE_K_V16_PER_MMA

/////////////////////
// tile n

#undef TILE_N_V1_PER_CTA
#undef TILE_N_V2_PER_CTA
#undef TILE_N_V4_PER_CTA
#undef TILE_N_V8_PER_CTA

#undef TILE_N_V1_PER_WARP
#undef TILE_N_V2_PER_WARP
#undef TILE_N_V4_PER_WARP
#undef TILE_N_V8_PER_WARP

#undef TILE_N_V1_PER_THD
#undef TILE_N_V2_PER_THD
#undef TILE_N_V4_PER_THD
#undef TILE_N_V8_PER_THD

#undef TILE_N_V1_PER_MMA
#undef TILE_N_V2_PER_MMA
#undef TILE_N_V4_PER_MMA
#undef TILE_N_V8_PER_MMA

#undef TILE_N_V1_PER_STEP
#undef TILE_N_V2_PER_STEP
#undef TILE_N_V4_PER_STEP
#undef TILE_N_V8_PER_STEP

////////////////////////////////////////
// main loop macros
////////////////////////////////////////

#undef C_ITEMS_PER_THD
#undef Cv4_ITEMS_PER_THD

////////////////////////////////////////
// load A and B from device memory macros
////////////////////////////////////////

#undef REG_dAv1_SIZE
#undef REG_dAv2_SIZE
#undef REG_dAv4_SIZE

#undef REG_dBv1_SIZE
#undef REG_dBv2_SIZE
#undef REG_dBv4_SIZE

#undef READ_dAv1_STEPS
#undef READ_dAv2_STEPS
#undef READ_dAv4_STEPS

#undef READ_dBv1_STEPS
#undef READ_dBv2_STEPS
#undef READ_dBv4_STEPS

////////////////////////////////////////
// shared memory size macros
////////////////////////////////////////

#undef SM_IN_ID_SIZE
#undef SM_OFF_ID_SIZE

////////////////////////////////////////
// mma macros
////////////////////////////////////////

#undef MMA_INST_OPCODE
#undef MMA_INST

#undef MMA_INST_1INT_ASCEND1
#undef MMA_INST_1INT_ASCEND2
#undef MMA_INST_1INT_ASCEND4
#undef MMA_INST_1INT_ASCEND8
#undef MMA_INST_1INT_ASCEND16

#undef MMA_INST_1INT_DESCEND1
#undef MMA_INST_1INT_DESCEND2
#undef MMA_INST_1INT_DESCEND4
#undef MMA_INST_1INT_DESCEND8
#undef MMA_INST_1INT_DESCEND16

#undef MMA_INST_1INT_1x1
#undef MMA_INST_1INT_1x2
#undef MMA_INST_1INT_1x4
#undef MMA_INST_1INT_1x8

#undef MMA_INST_1INT_2x1
#undef MMA_INST_1INT_2x2
#undef MMA_INST_1INT_2x4
#undef MMA_INST_1INT_2x8

#undef MMA_INST_1INT_4x1
#undef MMA_INST_1INT_4x2
#undef MMA_INST_1INT_4x4
#undef MMA_INST_1INT_4x8

#undef MMA_INST_1INT_8x1
#undef MMA_INST_1INT_8x2
#undef MMA_INST_1INT_8x4
#undef MMA_INST_1INT_8x8

#undef MMA_INST_1INT_16x1
#undef MMA_INST_1INT_16x2
#undef MMA_INST_1INT_16x4

#undef MMA_INST_2INT_ASCEND1
#undef MMA_INST_2INT_ASCEND2
#undef MMA_INST_2INT_ASCEND4
#undef MMA_INST_2INT_ASCEND8
#undef MMA_INST_2INT_ASCEND16

#undef MMA_INST_2INT_DESCEND1
#undef MMA_INST_2INT_DESCEND2
#undef MMA_INST_2INT_DESCEND4
#undef MMA_INST_2INT_DESCEND8
#undef MMA_INST_2INT_DESCEND16

#undef MMA_INST_2INT_1x1
#undef MMA_INST_2INT_1x2
#undef MMA_INST_2INT_1x4
#undef MMA_INST_2INT_1x8

#undef MMA_INST_2INT_2x1
#undef MMA_INST_2INT_2x2
#undef MMA_INST_2INT_2x4
#undef MMA_INST_2INT_2x8

#undef MMA_INST_2INT_4x1
#undef MMA_INST_2INT_4x2
#undef MMA_INST_2INT_4x4
#undef MMA_INST_2INT_4x8

#undef MMA_INST_2INT_8x1
#undef MMA_INST_2INT_8x2
#undef MMA_INST_2INT_8x4
#undef MMA_INST_2INT_8x8

#undef MMA_INST_2INT_16x1
#undef MMA_INST_2INT_16x2
#undef MMA_INST_2INT_16x4

#undef MMA_INST_4INT_ASCEND1
#undef MMA_INST_4INT_ASCEND2
#undef MMA_INST_4INT_ASCEND4
#undef MMA_INST_4INT_ASCEND8
#undef MMA_INST_4INT_ASCEND16
                 
#undef MMA_INST_4INT_DESCEND1
#undef MMA_INST_4INT_DESCEND2
#undef MMA_INST_4INT_DESCEND4
#undef MMA_INST_4INT_DESCEND8
#undef MMA_INST_4INT_DESCEND16
                 
#undef MMA_INST_4INT_1x1
#undef MMA_INST_4INT_1x2
#undef MMA_INST_4INT_1x4
#undef MMA_INST_4INT_1x8
                 
#undef MMA_INST_4INT_2x1
#undef MMA_INST_4INT_2x2
#undef MMA_INST_4INT_2x4
#undef MMA_INST_4INT_2x8
                 
#undef MMA_INST_4INT_4x1
#undef MMA_INST_4INT_4x2
#undef MMA_INST_4INT_4x4
#undef MMA_INST_4INT_4x8
                 
#undef MMA_INST_4INT_8x1
#undef MMA_INST_4INT_8x2
#undef MMA_INST_4INT_8x4
#undef MMA_INST_4INT_8x8

#undef MMA_INST_4INT_16x1
#undef MMA_INST_4INT_16x2
#undef MMA_INST_4INT_16x4

#undef MMA_INSTS

/////////////////////////////////////////////////////
// common load global memory macros
/////////////////////////////////////////////////////

//////////////////////////
// load dA
//////////////////////////

#undef LOAD_dAv1_SIZE_16TH
#undef LOAD_dAv1_SIZE_8TH
#undef LOAD_dAv1_SIZE_QTR
#undef LOAD_dAv1_SIZE_HALF
#undef LOAD_dAv1_SIZE1
#undef LOAD_dAv1_SIZE2
#undef LOAD_dAv1_SIZE4
#undef LOAD_dAv1_SIZE8
#undef LOAD_dAv1_SIZE16

#undef LOAD_dAv2_SIZE_16TH
#undef LOAD_dAv2_SIZE_8TH
#undef LOAD_dAv2_SIZE_QTR
#undef LOAD_dAv2_SIZE_HALF
#undef LOAD_dAv2_SIZE1
#undef LOAD_dAv2_SIZE2
#undef LOAD_dAv2_SIZE4
#undef LOAD_dAv2_SIZE8
#undef LOAD_dAv2_SIZE16

#undef LOAD_dAv4_SIZE_16TH
#undef LOAD_dAv4_SIZE_8TH
#undef LOAD_dAv4_SIZE_QTR
#undef LOAD_dAv4_SIZE_HALF
#undef LOAD_dAv4_SIZE1
#undef LOAD_dAv4_SIZE2
#undef LOAD_dAv4_SIZE4
#undef LOAD_dAv4_SIZE8
#undef LOAD_dAv4_SIZE16

#undef LOAD_dAv1
#undef LOAD_dAv2
#undef LOAD_dAv4

#undef SET_IN_Mv1_ID
#undef SET_IN_Kv16_OFF

//////////////////////////
// load dB
//////////////////////////

#undef LOAD_dBv1_SIZE_16TH
#undef LOAD_dBv1_SIZE_8TH
#undef LOAD_dBv1_SIZE_QTR
#undef LOAD_dBv1_SIZE_HALF
#undef LOAD_dBv1_SIZE1
#undef LOAD_dBv1_SIZE2
#undef LOAD_dBv1_SIZE4
#undef LOAD_dBv1_SIZE8

#undef LOAD_dBv2_SIZE_16TH
#undef LOAD_dBv2_SIZE_8TH
#undef LOAD_dBv2_SIZE_QTR
#undef LOAD_dBv2_SIZE_HALF
#undef LOAD_dBv2_SIZE1
#undef LOAD_dBv2_SIZE2
#undef LOAD_dBv2_SIZE4
#undef LOAD_dBv2_SIZE8

#undef LOAD_dBv4_SIZE_16TH
#undef LOAD_dBv4_SIZE_8TH
#undef LOAD_dBv4_SIZE_QTR
#undef LOAD_dBv4_SIZE_HALF
#undef LOAD_dBv4_SIZE1
#undef LOAD_dBv4_SIZE2
#undef LOAD_dBv4_SIZE4
#undef LOAD_dBv4_SIZE8

#undef LOAD_dBv1
#undef LOAD_dBv2
#undef LOAD_dBv4

#undef SET_dBv4_BOUND

/////////////////////////////////////////////////////
// precision half output
/////////////////////////////////////////////////////

#undef GET_DEQUANTSCALE
#undef DEQUANT_V2
#undef QUANT_V2

#undef OUTPUT_BY_HALF_X1
#undef OUTPUT_BY_HALF_X2

#undef PACK_V2
#undef PACK_V4

#undef ADD_BIAS_V2

#undef FUSE_RELU_V2
#undef FUSE_CLIP_V2
#undef FUSE_PRELU_V2

#undef FUSE_ELT_V2

#undef SET_CONCAT_OFF_V2


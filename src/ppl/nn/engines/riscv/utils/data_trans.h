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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_UTILS_DATA_TRANS_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_UTILS_DATA_TRANS_H_

#include <cstdint>

void N8cxToNdarrayFp16(const __fp16* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst);

void NdarrayToN8cxFp16(const __fp16* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst);

void N8cxToNdarrayFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst);

void N4cxToNdarrayFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst);

void NdarrayToN4cxFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst);

void NdarrayToN8cxFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst);

// TODO Optimize
void N8cxFp16ToNdarrayFp32(const __fp16* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst);

// TODO Optimize
void NdarrayFp32ToN8cxFp16(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst);

void N8cxFp16ToN4cxFp32(const __fp16* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst);

void N4cxFp32ToN8cxFp16(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst);

void N4cxFp32ToNdarrayFp16(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst);

#endif

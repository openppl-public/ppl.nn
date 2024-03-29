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

#include "ppl/nn/utils/compact_buffer_manager.h"
#include "gtest/gtest.h"
using namespace std;
using namespace ppl::nn;
using namespace ppl::common;

class TestAllocator final : public CompactAddrManager::Allocator {
public:
    TestAllocator(uint64_t alignment) : alignment_(alignment) {}
    pair<uintptr_t, uint64_t> Alloc(uint64_t size) {
        auto ptr = aligned_alloc(size, alignment_);
        if (!ptr) {
            return make_pair(UINTPTR_MAX, 0);
        }
        allocated_size_ += size;
        return make_pair((uintptr_t)ptr, size);
    }
    uint64_t GetAllocatedSize() const override {
        return allocated_size_;
    }
private:
    uint64_t allocated_size_ = 0;
    const uint64_t alignment_;
};

TEST(CompactBufferManagerTest, alloc_and_free) {
    const uint64_t bytes_needed = 1000;
    const uint64_t block_size = 1024;
    const uint64_t alignment = 128;

    TestAllocator ar(alignment);
    utils::CompactBufferManager mgr(&ar, alignment);
    BufferDesc buffer;
    auto status = mgr.Realloc(bytes_needed, &buffer);
    EXPECT_EQ(RC_SUCCESS, status);
    EXPECT_NE(nullptr, buffer.addr);
    EXPECT_LE(bytes_needed, buffer.desc);
    EXPECT_EQ(0, (uintptr_t)(buffer.addr) % alignment);
    mgr.Free(&buffer);
    EXPECT_EQ(block_size, mgr.GetBufferedBytes());
}

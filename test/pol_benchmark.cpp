// compile the file with the following:
// clang++ -O3 -mcpu=native -std=c++17 -I$(brew --prefix google-benchmark)/include -L$(brew --prefix google-benchmark)/lib -lbenchmark pol_benchmark.cpp aot_out/pol_1024_2.c aot_out/pol_2_1024.c -o aot_out/pol_benchmark
#include "benchmark/benchmark.h"

extern void pol_2_1024(double *__restrict__ out0_1024, double *__restrict__ in0_20);
extern void pol_1024_2(double *__restrict__ out0_1024,
                      double *__restrict__ in0_20);

#define no_optimize(ptr) asm volatile("" : : "r,m"(ptr[0]) : "memory");

static void bm_pol_2_1024(benchmark::State &state) {
  double out[1024], in[2048];

  for (auto _ : state) {
    pol_2_1024(out, in);
    no_optimize(out);
  }
}
BENCHMARK(bm_pol_2_1024);

static void bm_pol_1024_2(benchmark::State &state) {
  double out[1024], in[2048];

  for (auto _ : state) {
    pol_1024_2(out, in);
    no_optimize(out);
  }
}
BENCHMARK(bm_pol_1024_2);

BENCHMARK_MAIN();

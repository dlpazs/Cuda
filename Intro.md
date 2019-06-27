# Parallel programming
***
What are 3 traditional ways hardware designers make computers run faster?

- More work per clock cycle, **More Processors**, Faster Clocks

Modern GPUs have thousansds of Arithmetic Logical Units (that can simultanouesly run thousands of arithmetic operations), 
Hundreds of processors and tens of thousands of concurrent threads (e.g. up to 65,000 parallel pieces of work at the same time).

How to do parallel programming is the goal here

- modern processors are made from transistors and they get smaller over time, get faster, get more on a chip.
- as transistors improved processor designers then increased clock rate of processors running them faster and faster. 

Are processors getting faster because: we're clocking their transistors faster or we have more transistors per computation?

- The latter, clock rates are remaining constant and we are getting more transistors on a cpu

Why dont we increase clock speed? 
- though transistors get smaller and faster, running all these transistors increases heat and we can't cool them.
- Consequentially we can't keep building processors in the way we normally have by making it faster, since they overheat.
- instead they build smaller more efficient less power-hungry processors

What kind of processors will we build?

- CPU have complex control hardware and are flexible and performance but are expensive in terms of power
- GPU have less complex control structures and have more hardware for computation, more power efficient but more restrictive programming
model

Which techniques are computer designers using today to build more power efficient chips? 

- So processor clock speed is constant, complexity of control increases power, so the answer is more simpler processors the core of the GPU

So how do we design this :

Latency (time) (seconds) - latency is the amount of time to complete a task measured in seconds
Throughput (bandwidth) (stuff/time) (jobs/hour) - throughput tasks completed per unit time

CPU's optimize for latency they try to minimize the time elapsed for one particular task.
GPU's optimize for throughput. 

For instance, we have a car and a bus that travel 45km. The car carries 2 people and travels 200km/h and the bus 40 people at 50km/h.
The latency of the car is 22.5 which is good but the throughput is 0.089 which is bad. The bus' latency is 90 hours whilst the throughput
i.e. people/hours is much better at 0.45. 

Core GPU Design Tenets

1) lots of simple compute units, which trades off simple control for more compute
2) explicitly parallel programming model
3) optimize for throughput not latency

If you but an 8 core Ivy bridge processor from intel, each core has 8-wide AVX vector operations, each core supports 2 simultaneously
running threads (hyperthreading) and if you multiply those together 8 x 8 x 2 you get 128 way parallelism. If you make a program with
no parallelism at all you will make use of less than 1% of its capacity.

Heterogenous computers have 2 different processors, the CPU and GPU. If you write a plain C program it will only allow you to use the CPU
to run your programs. We use CUDA to write on the GPU. CUDA allows you to run both CPU and GPU with only one program. CUDA supports
numerous languages. Part of your CUDA is plain C and runs on the CPU i.e. the host, the other part runs on the GPU in parallel but with some
extensions and the term for the GPU is called Device. Then the CUDA compiler splits the program to run on the CPU and GPU. 
CUDA assumes the device is a co-processor to the host processor the CPU. It also assumes both have separate memories where they store
data. Both CPU and GPU have their own memory physical dedicated memory in the form of D-RAM, with GPUs being high performance.
The CPU is in charge and sends commands to the GPU. It's responsible for 1) moving data from CPU's memory to GPU's, 2) moving data back the
other way. In C, moving data from one place to another is called Memcpy so in CUDA it is called cudaMemcpy. 3) Allocating GPU memory
called cudaMalloc. 4) Invoking programs on the GPU that compute things in parallel, these things are called Kernels. So we say
the host launches kernels on the device.

So to recap, the GPU can do the following? Respond to CPU's request to send data from GPU to CPU. Respond to CPU request to receive
data from CPU to GPU. Compute kernel's launched by CPU. 

A typical GPU program looks like this:

1) CPU allocates storage on GPU, cudaMalloc
2) CPU copies input data from CPU to GPU, cudaMemcpy
3) CPU launches kernel(s) on GPU to process the data, kernal launch
4) CPU copies results back to CPU from GPU, cudaMemcpy
***
## Defining the GPU computation

The computation is a series of one or more kernels. The GPU has lots of parallel computation units. The **big core idea** you write
what looks like a serial program, it looks like it runs on one thread. The GPU will then run that program on many threads. 

What is the GPU good at? 

1) Efiiciently launching a large number of threads
2) Running a large number of threads in parallel

Simple Example:

```
In: Float Array [0, 1, 2, ... , 63]
Out: Float Array [0, 1, 2 ... , 63]

for (i = 0; i < 64; i++) {
  out[i] = in[i] * in[i];
}
```

1) In the program above we only have one thread of execution ("thread" = "one independent path of execution through the code")
2) No explicit parallelism

The above will do 64 multiplications, and say it takes 2 ns it will take 128 ns since we do it serially.

The CPu will allocate memory, copy data to/from the GPU and launch the kernel. The kernel has to specifiy the degree of paralleism.
The kernel on the gpu is then `Out = in * in` but says nothing about the degree of parallelism. 

`CPU code: Square Kernel<<<64>>>(outArray, inArray)` 

Launch a square kernel, launch 64 threads and their arguments are an output and input array.
What good is it running 64 instances of the same program? The CPU launches 64 threads, each thread knows which it is and 
each is indexed. You can assign thread N to work the Nth element of the array. 

How many multiplications will this new code perform? 64
Assume each multiplication takes 10ns how long does it take for the entire computation? 10ns

This further demonstrates the throughput versus latency debate. 

```
int main(int argc, char ** argv) {
  const int ARRAY_SIZE = 64;
  const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
  
  //generate the input array on the host
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) {
    h_in[i] = float(i);
  }
  float h_out[ARRAY_SIZE];
  
  //declare GPU memory pointers
  float * d_in;
  float * d_out;
  //to tell cuda your data is on the GPU not CPU do...
  //allocate GPU memory
  cudaMalloc((void **) &d_in, ARRAY_BYTES);//takes in a pointer and the number of bytes to allocate
  cudaMalloc((void **) &d_out, ARRAY_BYTES);// a normal Malloc would allocate data on the CPU
  
  //transfer the array to the GPU
  cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpy???);//
  
  //launch the kernel
  square<<<1, ARRAY_SIZE>>>(d_out, d_in);
  
  //copy back the result array to the CPU
  cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpy???);
  
  //print out the resulting array
  for (int i = 0; i < ARRAY_Size; i++) {
      
  }
}
```

The convention is `d` is for device and `h` for host. The most common error is trying to access data on CPU from GPU or vice versa.
If your accessing data on your CPU your pointer needs to point to something in CPU memory.

...34





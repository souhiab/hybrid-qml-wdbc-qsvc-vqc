// Single-threaded complex128 gate evolution. No fast-math, FMA or precision changes.
extern "C" __declspec(dllexport) int evolve(double* states, int rows, int dim,
 int nops, const int* kind, const int* qubit, const int* target,
 const double* cosine, const double* sine) {
 if (!states || rows<1 || dim<2 || (dim&(dim-1))) return 1;
 for (int row=0;row<rows;++row) {
  double* state=states+((__int64)row*dim*2);
  for (int op=0;op<nops;++op) {
   int stride=1<<qubit[op];
   if (kind[op]==0) {
    double c=cosine[op],s=sine[op];
    for (int block=0;block<dim;block+=2*stride) {
     for (int j=0;j<stride;++j) {
      int a=2*(block+j),b=2*(block+j+stride);
      double ar=state[a],ai=state[a+1],br=state[b],bi=state[b+1];
      state[a]=c*ar-s*br;state[a+1]=c*ai-s*bi;
      state[b]=s*ar+c*br;state[b+1]=s*ai+c*bi;
     }
    }
   } else {
    int bit=1<<target[op];
    for (int block=0;block<dim;block+=2*bit) {
     for (int j=0;j<bit;++j) {
      int index=block+j;
      if (index&stride) {
       int a=2*index,b=2*(index+bit);
       double real=state[a],imag=state[a+1];
       state[a]=state[b];state[a+1]=state[b+1];
       state[b]=real;state[b+1]=imag;
      }
     }
    }
   }
  }
 }
 return 0;
}

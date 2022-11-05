__kernel void logmap(__global float* v,
                     __global read_only float* a,
                     __global read_only float* b,
                     __global read_only int* o,
                     read_only int l,
                     read_only int n,
                     read_only int width,
                     __global float* d)
{
	int j = get_global_id(0);
  int i = get_global_id(1);
  int loc = width*i+j;

  float eps = 0.000001;
  float cv = v[loc];
  for(int it = 0; it < n; it++){
    int co = it%l;
    if(o[co]==0){
      float div = a[j]-2*a[j]*cv;
      cv = a[j]*cv*(1-cv);
      if(it==0)continue;
      div = fabs(div)+eps;
      div = log(div);
      d[loc]+=div;
    }else{
      float div = b[i]-2*b[i]*cv;
      cv = b[i]*cv*(1-cv);
      if(it==0)continue;
      div = fabs(div)+eps;
      div = log(div);
      d[loc]+=div;
    }
  }
  d[loc]= d[loc]/n;
}

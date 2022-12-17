__kernel void logmap(__global float* v,
                     __global float* a,
                     __global float* b,
                     __global int* o,
                     int l,
                     int n,
                     int width,
                     __global float* d,
		     int is_last,
		     int r)
{
  int i = get_global_id(0);//First dim is c_width
  int j = get_global_id(1);//Second dim is c_height
  int loc = width*j+i;     //Location in flat array is c_height * width + c_width


  float eps = 0.00;
  float div = 0.0;
  float cv = v[loc];
  float dsum = 0.0;

  int act_n = n + ((-n)%l);//Make sure multiple of seq length
  for(int it = 0; it < act_n; it++){
    //Get which coefficient
    int co = it%l;

    //Get next value
    if(o[co]==0){
      cv = a[j]*cv*(1-cv);
      div = a[j]-2*a[j]*cv;
    }else{
      cv = b[i]*cv*(1-cv);
      div = b[i]-2*b[i]*cv;
    }
    
    //Get Current val for exponent
    div = fabs(div)+eps;
    div = log(div);
    dsum+=div;
  }
  //Save to array
  v[loc] = cv;
  d[loc] += dsum/act_n; 
  if(is_last!=0) d[loc] /= r;
}

#include <stdio.h>
#include <stdint.h>
static void TransformWHT(const int16_t* in, int16_t* out){
  int tmp[16]; int i;
  for(i=0;i<4;++i){
    const int a0=in[0+i]+in[12+i]; const int a1=in[4+i]+in[8+i];
    const int a2=in[4+i]-in[8+i]; const int a3=in[0+i]-in[12+i];
    tmp[0+i]=a0+a1; tmp[8+i]=a0-a1; tmp[4+i]=a3+a2; tmp[12+i]=a3-a2;
  }
  for(i=0;i<4;++i){
    const int dc=tmp[0+i*4]+3;
    const int a0=dc+tmp[3+i*4]; const int a1=tmp[1+i*4]+tmp[2+i*4];
    const int a2=tmp[1+i*4]-tmp[2+i*4]; const int a3=dc-tmp[3+i*4];
    out[0]=(a0+a1)>>3; out[16]=(a3+a2)>>3; out[32]=(a0-a1)>>3; out[48]=(a3-a2)>>3; out+=64;
  }
}
int main(void){
  int16_t in[16], out[256]; int i;
  for(i=0;i<16;++i){int v; if(scanf("%d",&v)!=1) return 1; in[i]=(int16_t)v;}
  TransformWHT(in, out);
  for(i=0;i<256;i+=16)printf("%d ",out[i]);
  printf("\n");
  return 0;
}

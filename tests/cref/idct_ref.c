#include <stdio.h>
#include <stdint.h>
#define BPS 32
static uint8_t clip_8b(int v){return (!(v&~0xff))?v:(v<0)?0:255;}
#define STORE(x,y,v) dst[(x)+(y)*BPS] = clip_8b(dst[(x)+(y)*BPS] + ((v)>>3))
#define M1(a) ((((a)*20091)>>16)+(a))
#define M2(a) (((a)*35468)>>16)
static void ITransformOne(const int16_t* in, uint8_t* dst){
  int C[16], *tmp = C; int i;
  for(i=0;i<4;++i){
    const int a=in[0]+in[8]; const int b=in[0]-in[8];
    const int c=M2(in[4])-M1(in[12]); const int d=M1(in[4])+M2(in[12]);
    tmp[0]=a+d; tmp[1]=b+c; tmp[2]=b-c; tmp[3]=a-d; tmp+=4; in++;
  }
  tmp=C;
  for(i=0;i<4;++i){
    const int dc=tmp[0]+4;
    const int a=dc+tmp[8]; const int b=dc-tmp[8];
    const int c=M2(tmp[4])-M1(tmp[12]); const int d=M1(tmp[4])+M2(tmp[12]);
    STORE(0,i,a+d); STORE(1,i,b+c); STORE(2,i,b-c); STORE(3,i,a-d);
    tmp++;
  }
}
int main(void){
  /* stdin: 16 dequant coeffs (slot order) then 16 pred values raster */
  int16_t in[16];
  uint8_t buf[32*32];
  int i;
  for(i=0;i<16;++i){int v; if(scanf("%d",&v)!=1) return 1; in[i]=(int16_t)v;}
  for(i=0;i<32*32;++i) buf[i]=0;
  /* pred at dst area (2,2)..(5,5) */
  for(i=0;i<16;++i){int v; if(scanf("%d",&v)!=1) return 1; buf[2 + (i&3) + ((i>>2)+2)*BPS] = (uint8_t)v;}
  ITransformOne(in, buf + 2*BPS + 2);
  for(i=0;i<4;++i) printf("%d %d %d %d ", buf[2+(i+2)*BPS], buf[3+(i+2)*BPS], buf[4+(i+2)*BPS], buf[5+(i+2)*BPS]);
  printf("\n");
  return 0;
}

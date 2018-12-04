#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define num 2 + 1
#define num_tri 1000000

double sigmoid(int i,double sum[num]);
double energy(void);
double SimultaneousEqu(double x[num][num]);
double ConstantValue(void);
double ObtaineTheta(int n,int m,double c);
double ObtaineWeight(int i,int j,int n,int m,double theta_ij,double theta_nm,double c);
void initialization(void);
void CountState(double x[num]);
void Probability(void);

double theta[num]; //閾値
double w[num][num][num][num];//重み
double x[num][num];//後で変える//最初はダミーニューロン
double sum[num]; //重み付け総和
double Pr_x[16]; //出現確率

int main(void){
    static double sum[num];
    static double E; //エネルギー関数
    static double const_value;
    int i,j,n,m;

    const_value = ConstantValue();

    for(i=0;i<num;i++){
        for(j=0;j<num;j++){
            for(n=0;n<num;n++){
                for(m=0;m<num;m++){
                    if(i==j==n==m){
                        w[i][j][n][m] = 0.0;
                    }
                    else if(i==0 && j==0){
                        w[i][j][n][m] = -ObtaineTheta(n,m,const_value);
                    }
                    else if(n==0 && m==0){
                        w[i][j][n][m] = -ObtaineTheta(i,j,const_value);
                    }
                    else{
                        w[i][j][n][m] = ObtaineWeight(i,j,n,m,ObtaineTheta(i,j,const_value),ObtaineTheta(n,m,const_value),const_value);
                    }
                    //printf("w%d%d%d%d:%lf\n",i,j,n,m,w[i][j][n][m]);
                }
            }
        }
    }
    initialization();
    for(i=0;i<num;i++){

        for(j=0;j<num;j++){
            if(i!=0&&j!=0){
                printf("x%d%d:%lf",i,j,x[i][j]);
                printf(",");
                if(j == (num - 1)){
                    printf("\n");
                }
            }
        }
    }
    
    
}

double sigmoid(int i,double sum[]){
    double p;
    double a = 1.5;
    p = 1 / (1 + exp(-a * sum[i]));
    
    return p;
}


double energy(void){
    double e = 0.0;
    int i,j;
    
    for(i=0;i<num;i++){
        for(j=0;j<num;j++){
            //e += w[i][j] * x[i] * x[j];
        }
    }
    return e;
}
//n,m番目の重みを求める、必要な値は予め求めておく
double ObtaineWeight(int i,int j,int n,int m,double theta_ij,double theta_nm,double C){
    int a,b,c,d;
    
    for(a=0;a<num;a++){
        for(b=0;b<num;b++){
            if((a==0&&b==0)&&(a==i&&b==j)&&(a==n&&b==m)){
                x[a][b] = 1.0;
            }
            else if(a==0&&b==0){
                x[a][b] = 0.0;
            }
            else{
                x[a][b] = 0.0;
            }
        }
    }
    return -SimultaneousEqu(x) + theta_ij + theta_nm + C;
}

//n番目のthetaを求める、cは定数の値を予め求める
double ObtaineTheta(int n,int m,double c){
    int i,j;
    double E_n = 0.0;
    
    for(i=0;i<num;i++){

        for(j=0;j<num;j++){
            if(i == n || j == m){
                x[i][j] = 1.0;
            }
            else{
                x[i][j] = 0.0;
            }
        }
    }
    E_n = SimultaneousEqu(x) - c;
    return E_n;
}
//定数を求める
double ConstantValue(void){
    int i,j;
    double C = 0.0;
    
    for(i=0;i<num;i++){
        for(j=0;j<num;j++){
            x[i][j] = 0.0;
        }
    }
    C = SimultaneousEqu(x);
    //printf("%lf\n",C);
    return C;
}

double SimultaneousEqu(double x[num][num]){
    double E = 0.0,E1 = 0.0,E2 = 0.0;
    double sum1 = 0.0,sum2 = 0.0;
    int i,j;
    
    for(i=0;i<num;i++){
        
        for(j=0;j<num;j++){
            if(i != 0 && j != 0){
                sum1 += x[i][j];
                sum2 += x[j][i];
                //printf("(%d,%d):%lf\n",i,j,x[i][j]);
                if(j == num-1){
                    E1 += (sum1-1.0) * (sum1-1.0);
                    E2 += (sum2-1.0) * (sum2-1.0);
                }
            }
        }
        sum1 = 0.0;
        sum2 = 0.0;
    }
    E = E1 + E2;
    return E;
}

void initialization(void){
    double ini_x[num][num] = { 
    {1.0,0.0,0.0},
    {0.0,1.0,0.0},
    {0.0,0.0,1.0}
};//最初の１はダミーニューロン
    int i,j;
    for(i=0;i<num;i++){

        for(j=0;j<num;j++){
            x[i][j] = ini_x[i][j];
        }
    }
    
}

void CountState(double x[num]){
    char str[100];
    int i;

    for(i=0;i<num;i++){
        if(i !=0 ){
            if(x[i] == 1.0){
                str[i-1] = '1';
            }
            else if(x[i] == 0.0){
                str[i-1] = '0';
            }
            else{
                str[i-1] = 'n';
            }
        }
    }
    //printf("%s\n",str);

    if(!strcmp(str,"0000")){
        Pr_x[0] += 1.0; 
    }
    else if(!strcmp(str,"0001")){
        Pr_x[1] += 1.0; 
    }
    else if(!strcmp(str,"0010")){
        Pr_x[2] += 1.0; 
    }
    else if(!strcmp(str,"0011")){
        Pr_x[3] += 1.0; 
    }
    else if(!strcmp(str,"0100")){
        Pr_x[4] += 1.0; 
    }
    else if(!strcmp(str,"0101")){
        Pr_x[5] += 1.0; 
    }
    else if(!strcmp(str,"0110")){
        Pr_x[6] += 1.0; 
    }
    else if(!strcmp(str,"0111")){
        Pr_x[7] += 1.0; 
    }
    else if(!strcmp(str,"1000")){
        Pr_x[8] += 1.0; 
    }
    else if(!strcmp(str,"1001")){
        Pr_x[9] += 1.0; 
    }
    else if(!strcmp(str,"1010")){
        Pr_x[10] += 1.0; 
    }
    else if(!strcmp(str,"1011")){
        Pr_x[11] += 1.0; 
    }
    else if(!strcmp(str,"1100")){
        Pr_x[12] += 1.0; 
    }
    else if(!strcmp(str,"1101")){
        Pr_x[13] += 1.0; 
    }
    else if(!strcmp(str,"1110")){
        Pr_x[14] += 1.0; 
    }
    else if(!strcmp(str,"1111")){
        Pr_x[15] += 1.0; 
    }
}

void Probability(void){
    int i;

    for(i=0;i<16;i++){
        printf("Pr_x[%d] : %lf \n",i,Pr_x[i] / ((num-1) * num_tri));
    }
}


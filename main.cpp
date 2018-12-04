#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define num 4 + 1
#define num_tri 1000000
int N = pow(2,16);

double sigmoid(int i,double sum[num]);
double energy(void);
double SimultaneousEqu(double x[num][num]);
double ConstantValue(void);
double ObtaineTheta(int n,int m,double c);
double ObtaineWeight(int i,int j,int n,int m,double theta_ij,double theta_nm,double c);
void initialization(void);
void CountState(int i,int j,double x[num][num]);
void Probability(void);
void decisiveRNN(void);
void probabilisticRNN(void);

bool probabilistic = false;

double theta[num]; //閾値
double w[num][num][num][num];//重み
double x[num][num];//後で変える//最初はダミーニューロン
//double Pr_x[N]; //出現確率
double const_value;
int kaisu[num][num];

int main(void){
    static double E; //エネルギー関数
    int i,j,n,m;
    int NumOfState;
    NumOfState = pow(2,num-1);
    double sum[NumOfState]; //重み付け総和

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
    //CountState(x);
    decisiveRNN();
    //probabilisticRNN();

}

void decisiveRNN(void){
    int i,j,k,l,n,m;
    double e;
    double sum[num];
    bool done = false;

    for(l=0;l<100;l++){
        initialization();
        done = false;
        for(k=0;k<10;k++){
            for(i=0;i<num;i++){
                for(j=0;j<num;j++){
                    for(n=0;n<num;n++){
                        for(m=0;m<num;m++){
                            sum[j] += w[n][m][i][j] * x[n][m];
                        }
                    }
                    if(i != 0 && j != 0){
                        if(sum[j]>=0){
                            x[i][j] = 1.0;
                        }
                        else{
                            x[i][j] = 0.0;
                        }
                    }
                    e = energy();
                    //printf("%lf\n",e);
                    
                    if(e<=6){
                        //CountState(i,j,x);
                        //done = true;
                    }
                    if(done){
                        e=0;
                        break;
                    }
                    sum[j] = 0.0;
                }
                if(done){
                    break;
                }
            }
            if(done){
                break;
            }
            if(k==9){
                for(i=0;i<num;i++){
                    for(j=0;j<num;j++){
                        if(i!=0&&j!=0){
                            printf("x%d%d:%lf",i,j,x[i][j]);
                                printf(",");
                                if(j == (num - 1)){
                                    printf("\n");
                                    if(i == (num-1)){
                                        printf("=====\n");
                                    }
                                }
                        }
                    }
                }
                CountState(1,1,x);
            }
        }
    }

}

void probabilisticRNN(void){
    int i,j,n,m;
    double sum[num];
    double e;

    for(int l=0;l<10;l++){
        initialization();
        for(int k=0;k<10;k++){
            for(i=0;i<num;i++){
                for(j=0;j<num;j++){
                    for(n=0;n<num;n++){
                        for(m=0;m<num;m++){
                            sum[j] += w[n][m][i][j] * x[n][m];
                        }
                    }
                    if(i != 0 && j != 0){
                        if(rand() < RAND_MAX * sigmoid(j,sum)){
                            x[i][j] = 1.0;
                        }
                        else{
                            x[i][j] = 0.0;
                        }
                    }
                    //確認用
                    #if 0
                    if(i!=0&&j!=0){
                    printf("x%d%d:%lf",i,j,x[i][j]);
                        printf(",");
                        if(j == (num - 1)){
                            printf("\n");
                            if(i == (num-1)){
                                printf("-----\n");
                            }
                        }
                    }
                    #endif
                    sum[j] = 0.0;
                }
            }
        }
        e = energy() - const_value;
        //printf("%lf\n",e);
        e = 0;
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
    int i,j,n,m;
    
    for(i=0;i<num;i++){
        for(j=0;j<num;j++){
            for(n=0;n<num;n++){
                for(m=0;m<num;m++){
                    e += w[i][j][n][m] * x[i][j] * x[n][m];
                }
            }
        }
    }
    return -0.5*e + const_value;
}
//n,m番目の重みを求める、必要な値は予め求めておく
double ObtaineWeight(int i,int j,int n,int m,double theta_ij,double theta_nm,double C){
    int a,b,c,d;
    
    for(a=0;a<num;a++){
        for(b=0;b<num;b++){
            if((a==0&&b==0)||(a==i&&b==j)||(a==n&&b==m)){
                x[a][b] = 1.0;
            }
            else if(a==0||b==0){
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
            if(i == n && j == m){
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
    int samples = 100;
    double n=0.0;
    //double ini_x[samples][num][num];
    double ini_x[num][num];/* = { 
    {1.0,0.0,0.0},
    {0.0,1.0,0.0},
    {0.0,0.0,1.0}
};*///最初の１はダミーニューロン

    int i,j,k;
    
    for(i=0;i<num;i++){

        for(j=0;j<num;j++){
            if(i==0&&j==0){
                ini_x[i][j] = 1.0;
            }
            else if(i==0||j==0){
                ini_x[i][j] = 0.0;
            }
            else{
                ini_x[i][j] = (double)(rand() % 2);
            }
            x[i][j] = ini_x[i][j];
        }
    }
    #if 0
    for(i=0;i<num;i++){
        for(j=0;j<num;j++){
            if(i!=0&&j!=0){
                if(j==1){
                    printf("ini->");
                }
                printf("x%d%d:%lf",i,j,x[i][j]);
                    printf(",");
                    if(j == (num - 1)){
                        printf("\n");
                        if(i == (num-1)){
                            printf("=====\n");
                        }
                    }
            }
        }
    }
    #endif
    
}

void CountState(int i,int j,double x[num][num]){
    int k,l,s=0;
    char str[num*num+10];
    char c[N][num*num+10];

    for(k=0;k<num;k++){
        for(l=0;l<num;l++){
            if(k!=0&&l!=0){
                if(x[k][l] == 1.0){
                    str[s] = '1';
                }
                else if(x[k][l] == 0.0){
                    str[s] = '0';
                }
                else{
                    str[s] = 'n';
                }
                s++;
            }
        }
    }
    //printf("%s\n",str);

    for(k=0;k<num;k++){
        for(l=0;l<num;l++){
            if(k==i&&l==j){
                kaisu[k][l] += 1;
            }
            #if 0
            if(k!=0&&l!=0){
                printf("x%d%d:%lf",k,l,x[k][l]);
                printf(",");
                if(l == (num-1)){
                    printf("\n");
                    if(k == (num-1)){
                        printf("-----\n");
                    }
                }
            }
            #endif
        }
    }
}

void Probability(void){
    int i;

    for(i=0;i<16;i++){
        //printf("Pr_x[%d] : %lf \n",i,Pr_x[i] / ((num-1) * num_tri));
    }
}


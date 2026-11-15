# include <stdio.h>
# include <math.h>
# include <vector>
using namespace std;
typedef vector<double> Data;
int main(int argc,char **argv)
{
	vector<Data> xdata;
	Data ydata,mindata,maxdata;
	int d,n;
	const int divisions=5;
	//argv[1] filename
	//argv[2] mapfile
	if(argc==3)
	{
		//we have map file
		FILE *fp=fopen(argv[1],"r");
		if(!fp)
		{
			printf("can not open %s\n",argv[1]);
			return 1;
		}
		fscanf(fp,"%d",&d);
		fscanf(fp,"%d",&n);
		xdata.resize(n);
		ydata.resize(n);
		for(int i=0;i<n;i++)
		{
			xdata[i].resize(d);
			for(int j=0;j<d;j++)
				fscanf(fp,"%lf",&xdata[i][j]);
			fscanf(fp,"%lf",&ydata[i]);
		}
		fclose(fp);
		fp=fopen(argv[2],"r");
		if(!fp)
		{
			printf("can not open %s\n",argv[2]);
			return 1;
		}
		fscanf(fp,"%d",&d);
		fscanf(fp,"%d",&divisions);
		mindata.resize(d);
		maxdata.resize(d);
		for(int i=0;i<d;i++) 
		{
			fscanf(fp,"%lf %lf",&mindata[i],&maxdata[i]);
		}

		fclose(fp);
		for(int i=0;i<d;i++) 
			printf("f%d,",i);
		printf("class\n");
		for(int i=0;i<n;i++)
		{

			for(int j=0;j<d;j++)
			{
				double xstart=mindata[j];
				bool found=false;
				for(int k=0;k<divisions;k++)
				{
					double xend=xstart+(maxdata[j]-mindata[j])/divisions;
					if(xdata[i][j]>=xstart && xdata[i][j]<=xend) 
					{
						printf("%.2lf,",xstart);
						found=true;
						break;
					}
					xstart=xend;
				}	
				if(!found) printf("%.2lf,",xstart);
			}
			printf("%.2lf\n",ydata[i]);
		}
	}
	else
	if(argc==2)
	{
		FILE *fp=fopen(argv[1],"r");
		if(!fp)
		{
			printf("can not open %s\n",argv[1]);
			return 1;
		}
		fscanf(fp,"%d",&d);
		fscanf(fp,"%d",&n);
		xdata.resize(n);
		ydata.resize(n);
		mindata.resize(d);
		maxdata.resize(d);
		for(int i=0;i<n;i++)
		{
			xdata[i].resize(d);
			for(int j=0;j<d;j++)
			{
				fscanf(fp,"%lf",&xdata[i][j]);
				if(i==0 || xdata[i][j]<mindata[j]) mindata[j]=xdata[i][j];
				if(i==0 || xdata[i][j]>maxdata[j]) maxdata[j]=xdata[i][j];	
			}
			fscanf(fp,"%lf",&ydata[i]);
		}
		fclose(fp);
		for(int i=0;i<d;i++) 
			printf("f%d,",i);
		printf("class\n");
		for(int i=0;i<n;i++)
		{

			for(int j=0;j<d;j++)
			{
				double xstart=mindata[j];
				bool found=false;
				for(int k=0;k<divisions;k++)
				{
					double xend=xstart+(maxdata[j]-mindata[j])/divisions;
					if(xdata[i][j]>=xstart && xdata[i][j]<=xend) 
					{
						printf("%.2lf,",xstart);
						found=true;
						break;
					}
					xstart=xend;
				}	
				if(!found) printf("%.2lf,",xstart);
			}
			printf("%.2lf\n",ydata[i]);
		}
		fp=fopen("map.txt","w");
		fprintf(fp,"%d\n",d);
		fprintf(fp,"%d\n",divisions);
		for(int i=0;i<d;i++)
		{
			fprintf(fp,"%lf %lf\n",mindata[i],maxdata[i]);
		}
		fclose(fp);
	}
	else 
	printf("Invalid argument count\n");
	return 0;
}

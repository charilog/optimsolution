# include <stdio.h>
int main(int argc,char **argv)
{
	FILE *fp=fopen(argv[1],"r");
	while(1)
	{
		int c=fgetc(fp);
		if(c==-1) break;
		if(c!=13) printf("%c",c);
	}
	if(!fp) return 0;
	fclose(fp);
	return 0;
}

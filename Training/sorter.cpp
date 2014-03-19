//Sorts images as per labels

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <string>
#include <vector>
using namespace std;

int main()
{
    ifstream f1("labels.txt");
    
    //chdir("Images");
   // system("ls");
    string str;
    string label;
  //  system("pwd");
    while( !f1.eof() )
    {   
        
        f1 >> str;
        f1 >> label;
        
        
        string command = string("mv Images/")  + str + string(" ./") + label + string("/") + str;
        cout<<"\n" << command;
        system(command.c_str());
        
    }
    
    return 0;
}

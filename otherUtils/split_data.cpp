#include <bits/stdc++.h>
#include <unistd.h>

#define SPLIT_SIZE 10000

using namespace std;
typedef long long ll;

string intToStr(ll n){
  string s = "";
  while(n){
    ll aux = n%10;
    n /= 10;
    char c = '0'+aux;
    s = c+s;
  }

  return s;
}

int main(){
  char currentDir[FILENAME_MAX];
  getcwd(currentDir, sizeof(currentDir));

  string cwd = string(currentDir);

  cout << cwd << endl;

  string data_dir = cwd + "/all_files_with_abstract_titles.csv";
  ifstream data(data_dir.c_str());

  ll cnt = 0;
  ll block = 0;

  string headers, line;
  getline(data, headers);

  ofstream file;

  while(getline(data, line)){
    cout << line << endl;
    if(cnt % SPLIT_SIZE == 0){
      file.flush();
      file.close();

      string block_str = cwd + "/files_with_abstract_titles_block_" + intToStr(block) + ".csv";
      file.open(block_str.c_str());
      file << headers << "\n";
    }

    file << line << "\n";
  }

  if(cnt % SPLIT_SIZE != 0){
    file.flush();
    file.close();
  }

  return 0;
}

#include <bits/stdc++.h>
#include <filesystem>
#define SPLIT_SIZE 10000

using namespace std;
typedef long long ll;

int main(){
  ifstream data("/home/vyral/Documents/UConn/CrimProject/data/criminal_data_full.csv");

  filesystem::path cwd = filesystem::current_path();

  ll cnt = 0;
  ll block = 0;

  string headers, line;
  getline(data, headers);

  ofstream file;

  while(getline(data, line)){
    if(cnt % SPLIT_SIZE == 0){
      file.flush();
      file.close();

      string block_str = "files_with_abstract_titles_block_" + to_string(block) + ".csv";
      file.open(cwd / block_str);
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

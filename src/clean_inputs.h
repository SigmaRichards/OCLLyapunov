#include <regex>
#include <sstream>

int clean_int(char* optionarg){
  std::regex re("^\\d+$");
  std::cmatch m;
  bool t = std::regex_match(optionarg, m, re);
  if(t){
    return std::stoi(optionarg);
  }
  return -1;
}

std::vector<int> clean_inta(char* optionarg){
  std::vector<int> out;
  std::regex re("^(\\d+,)*\\d+$");
  std::cmatch m;
  bool t = std::regex_match(optionarg, m, re);
  if(t){
    std::stringstream s;
    std::string segment;
    s << optionarg;
    while(std::getline(s, segment, ','))
    {
       out.push_back(std::stoi(segment));
    }
  }
  return out;
}

float clean_float(char* optionarg){
  std::regex re("^\\d*\\.{0,1}\\d+$");
  std::cmatch m;
  bool t = std::regex_match(optionarg, m, re);
  if(t){
    return std::stof(optionarg);
  }
  return -1;
}

std::vector<float> clean_floata(char* optionarg){
  std::vector<float> out;
  std::regex re("^(\\d+(\\.\\d+){0,1},)*\\d+(\\.\\d+){0,1}$");
  std::cmatch m;
  bool t = std::regex_match(optionarg, m, re);
  if(t){
    std::stringstream s;
    std::string segment;
    s << optionarg;
    while(std::getline(s, segment, ','))
    {
       out.push_back(std::stof(segment));
    }
  }
  return out;
}

std::string clean_col(char* col){
  std::string out = "";
  std::regex re("^\\#[0-9a-fA-F]{6}$");
  std::cmatch m;
  bool t = std::regex_match(col, m, re);
  if(t){
    out = col;
  }
  return out;
}

std::vector<float> hex2col(std::string col){
  unsigned int r = std::stoul(col.substr(1,2), nullptr, 16);
  unsigned int g = std::stoul(col.substr(3,2), nullptr, 16);
  unsigned int b = std::stoul(col.substr(5,2), nullptr, 16);
  std::vector<float> out = {(float)(int)r/255,(float)(int)g/255,(float)(int)b/255};
  return out;
}

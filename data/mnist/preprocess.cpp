#include<fstream>
#include<iterator>
#include<sstream>
#include<string>
#include<vector>
using namespace std;

const int trainRow = 60000, testRow=10000;
const int col = 785, classNum = 10;

// Add 1 to col, col originally 784, now = 785, the last col valued 1

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

/* BIG SHOUTOUT to my hours of lazy mistake
  it turns out, MPI-IO is way better (and supposed) to read BINARY,
  and NOT TEXT, so yeah, we will rewrite these data into
  BINARY preprocessed data.
*/


void preprocess(string rawFileName, string xFileName, string yFileName){
	/*	We are going to read the raw file line by line
		and write each line (which means each row of data)
		into corresponding output file_x and file_y
	*/
	ifstream rawFile(rawFileName);
	ofstream xFile(xFileName, ios::trunc | ios::binary);
	ofstream yFile(yFileName, ios::trunc | ios::binary);
	string line;
	while (getline(rawFile, line)){
		istringstream stLine(line);

		// first number is the class
		float yArr[classNum];
    int yVal;
		for (int cn=0;cn<classNum;cn++){
			yArr[cn]=0;
		}
		stLine >> yVal;
		yArr[yVal]=1;

		// Then the rest is the x data
		float xArr[col], xVal;
		int xIdx;
		for (int c=0;c<col-1;c++){
			xArr[c]=0;
		}
    xArr[col-1] = 1;

		string xRaw;
		while(stLine >> xRaw){
			vector<string> xVec = split(xRaw, ':');
			xIdx = stoi(xVec[0]);
			xVal = stod(xVec[1]);
			xArr[xIdx]=xVal;
		}

		// Write to output files
		yFile.write((char*) &yArr, classNum*sizeof(float));
    xFile.write((char*) &xArr, col*sizeof(float));
	}

	rawFile.close();
	xFile.close();
	yFile.close();
}

int main(){
	// ios_base::sync_with_stdio(false);
	// Preprocess training
	preprocess("raw/training", "training/file_x.bin", "training/file_y.bin");
  preprocess("raw/test", "test/file_x.bin", "test/file_y.bin");
}

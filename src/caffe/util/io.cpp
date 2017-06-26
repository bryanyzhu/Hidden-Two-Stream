#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

bool ReadSegmentRGBToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum, bool is_color,
    const char* name_pattern ){
  cv::Mat cv_img;
  string* datum_string;
  char tmp[30];
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
      CV_LOAD_IMAGE_GRAYSCALE);
  for (int i = 0; i < offsets.size(); ++i){
    int offset = offsets[i];
    for (int file_id = 1; file_id < length+1; ++file_id){
      sprintf(tmp, name_pattern, int(file_id+offset));
      string filename_t = filename + "/" + tmp;
      cv::Mat cv_img_origin = cv::imread(filename_t, cv_read_flag);
      if (!cv_img_origin.data){
        LOG(ERROR) << "Could not load file " << filename;
        return false;
      }
      if (height > 0 && width > 0){
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
      }else{
        cv_img = cv_img_origin;
      }
      int num_channels = (is_color ? 3 : 1);
      if (file_id==1 && i==0){
        datum->set_channels(num_channels*length*offsets.size());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        datum->set_label(label);
        datum->clear_data();
        datum->clear_float_data();
        datum_string = datum->mutable_data();
      }
      if (is_color) {
          for (int c = 0; c < num_channels; ++c) {
            for (int h = 0; h < cv_img.rows; ++h) {
              for (int w = 0; w < cv_img.cols; ++w) {
                datum_string->push_back(
                  static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
              }
            }
          }
        } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
          for (int h = 0; h < cv_img.rows; ++h) {
            for (int w = 0; w < cv_img.cols; ++w) {
              datum_string->push_back(
                static_cast<char>(cv_img.at<uchar>(h, w)));
              }
            }
        }
    }
  }
  return true;
}

bool ReadSegmentFlowToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum,
    const char* name_pattern ){
  cv::Mat cv_img_x, cv_img_y;
  string* datum_string;
  char tmp[30];
  for (int i = 0; i < offsets.size(); ++i){
    int offset = offsets[i];
    for (int file_id = 1; file_id < length+1; ++file_id){
      sprintf(tmp,name_pattern, 'x', int(file_id+offset));
      string filename_x = filename + "/" + tmp;
      cv::Mat cv_img_origin_x = cv::imread(filename_x, CV_LOAD_IMAGE_GRAYSCALE);
      sprintf(tmp, name_pattern, 'y', int(file_id+offset));
      string filename_y = filename + "/" + tmp;
      cv::Mat cv_img_origin_y = cv::imread(filename_y, CV_LOAD_IMAGE_GRAYSCALE);
      if (!cv_img_origin_x.data || !cv_img_origin_y.data){
        LOG(ERROR) << "Could not load file " << filename_x << " or " << filename_y;
        return false;
      }
      if (height > 0 && width > 0){
        cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
        cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
      }else{
        cv_img_x = cv_img_origin_x;
        cv_img_y = cv_img_origin_y;
      }
      if (file_id==1 && i==0){
        int num_channels = 2;
        datum->set_channels(num_channels*length*offsets.size());
        datum->set_height(cv_img_x.rows);
        datum->set_width(cv_img_x.cols);
        datum->set_label(label);
        datum->clear_data();
        datum->clear_float_data();
        datum_string = datum->mutable_data();
      }
      for (int h = 0; h < cv_img_x.rows; ++h){
        for (int w = 0; w < cv_img_x.cols; ++w){
          datum_string->push_back(static_cast<char>(cv_img_x.at<uchar>(h,w)));
        }
      }
      for (int h = 0; h < cv_img_y.rows; ++h){
        for (int w = 0; w < cv_img_y.cols; ++w){
          datum_string->push_back(static_cast<char>(cv_img_y.at<uchar>(h,w)));
        }
      }
    }
  }
  return true;
}

bool ReadSegmentRGBFlowToDatum(const string& filename1, const string& filename2, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum){
  
  cv::Mat cv_img;    // for RGB image
  cv::Mat cv_img_x, cv_img_y;    // for flow images, two channel   
  string* datum_string;
  // You can define your own name pattern to match the name of your images and load the data.
  string rgb_name_pattern = "image_%04d.jpg";
  string flow_name_pattern = "flow_%c_%04d.jpg";
  char tmp[30];
  
  for (int i = 0; i < offsets.size(); ++i){
    int offset = offsets[i];

    // load RGB data
    for (int file_id = 1; file_id < length+1; ++file_id){
      sprintf(tmp, rgb_name_pattern.c_str(), int(file_id+offset));
      string filename_rgb = filename1 + "/" + tmp;
      // LOG(INFO) << "output rgb " << file_id << " : " << filename_rgb;
      cv::Mat cv_img_origin = cv::imread(filename_rgb, CV_LOAD_IMAGE_COLOR);
      if (!cv_img_origin.data){
        LOG(ERROR) << "Could not load file " << filename_rgb;
        return false;
      }
      if (height > 0 && width > 0){
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
      }else{
        cv_img = cv_img_origin;
      }

      int num_channels = 3;    // Default: the images are color images, 3-channel
      if (file_id==1 && i==0){
        datum->set_channels(num_channels*length*offsets.size() + (num_channels-1)*(length-1)*offsets.size());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        datum->set_label(label);
        datum->clear_data();
        datum->clear_float_data();
        datum_string = datum->mutable_data();
      }

      for (int c = 0; c < num_channels; ++c) {
        for (int h = 0; h < cv_img.rows; ++h) {
          for (int w = 0; w < cv_img.cols; ++w) {
            datum_string->push_back(
              static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
          }
        }
      }

    }

    // Load optical flow
    for (int file_id = 1; file_id < length; ++file_id){
      sprintf(tmp, flow_name_pattern.c_str(), 'x', int(file_id+offset));
      string filename_x = filename2 + "/" + tmp;
      // LOG(INFO) << "output flow " << file_id << " : " << filename_x;
      cv::Mat cv_img_origin_x = cv::imread(filename_x, CV_LOAD_IMAGE_GRAYSCALE);
      sprintf(tmp, flow_name_pattern.c_str(), 'y', int(file_id+offset));
      string filename_y = filename2 + "/" + tmp;
      cv::Mat cv_img_origin_y = cv::imread(filename_y, CV_LOAD_IMAGE_GRAYSCALE);
      if (!cv_img_origin_x.data || !cv_img_origin_y.data){
        LOG(ERROR) << "Could not load file " << filename_x << " or " << filename_y;
        return false;
      }

      if (height > 0 && width > 0){
        cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
        cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
      }else{
        cv_img_x = cv_img_origin_x;
        cv_img_y = cv_img_origin_y;
      }

      for (int h = 0; h < cv_img_x.rows; ++h){
        for (int w = 0; w < cv_img_x.cols; ++w){
          datum_string->push_back(static_cast<char>(cv_img_x.at<uchar>(h,w)));
        }
      }
      for (int h = 0; h < cv_img_y.rows; ++h){
        for (int w = 0; w < cv_img_y.cols; ++w){
          datum_string->push_back(static_cast<char>(cv_img_y.at<uchar>(h,w)));
        }
      }
    }

  }
  return true;
}

bool ReadSegmentMultiRGBToDatum(const string& filename, const int label,
    const vector<int> offsets, const int height, const int width, const int length, Datum* datum){
  
  cv::Mat cv_img;    // for RGB image
  string* datum_string;
  // You can define your own name pattern to match the name of your images and load the data.
  string rgb_name_pattern = "image_%04d.jpg";
  char tmp[30];
  
  for (int i = 0; i < offsets.size(); ++i){
    int offset = offsets[i];

    // load RGB data
    for (int file_id = 1; file_id < length+1; ++file_id){
      sprintf(tmp, rgb_name_pattern.c_str(), int(file_id+offset));
      string filename_rgb = filename + "/" + tmp;
      // LOG(INFO) << "output rgb " << file_id << " : " << filename_rgb;
      cv::Mat cv_img_origin = cv::imread(filename_rgb, CV_LOAD_IMAGE_COLOR);
      if (!cv_img_origin.data){
        LOG(ERROR) << "Could not load file " << filename_rgb;
        return false;
      }
      if (height > 0 && width > 0){
        cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
      }else{
        cv_img = cv_img_origin;
      }

      int num_channels = 3;    // Default: the images are color images, 3-channel
      if (file_id==1 && i==0){
        datum->set_channels(num_channels*length*offsets.size());
        datum->set_height(cv_img.rows);
        datum->set_width(cv_img.cols);
        datum->set_label(label);
        datum->clear_data();
        datum->clear_float_data();
        datum_string = datum->mutable_data();
      }

      for (int c = 0; c < num_channels; ++c) {
        for (int h = 0; h < cv_img.rows; ++h) {
          for (int w = 0; w < cv_img.cols; ++w) {
            datum_string->push_back(
              static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
          }
        }
      }

    }

  }
  return true;
}

#endif  // USE_OPENCV
}  // namespace caffe

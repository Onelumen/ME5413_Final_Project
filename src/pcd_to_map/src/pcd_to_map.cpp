#include <ros/ros.h>

#include <nav_msgs/GetMap.h>
#include <nav_msgs/OccupancyGrid.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/filters/conditional_removal.h> 
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>

std::string file_directory;
std::string file_name;
std::string pcd_file;

std::string map_topic_name;

const std::string pcd_format = ".pcd";

nav_msgs::OccupancyGrid map_topic_msg;
// minimum and maximum height
double thre_z_min = 0.3;
double thre_z_max = 2.0;
double map_resolution = 0.05;
int flag_pass_through = 0;
double thre_radius = 0.1;
// radius filter points threshold
int thres_point_count = 10;
double robot_reach_min = 0.1;
double robot_reach_max = 0.8;
// Minimum thickness to be considered an obstacle (thickness check)
double min_obstacle_height = 0.2;

pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_after_PassThrough_z(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_after_PassThrough_y(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_after_PassThrough_x(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr
    cloud_after_Radius(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr
    pcd_cloud(new pcl::PointCloud<pcl::PointXYZ>);

// pass through filter
void PassThroughFilter(const double &thre_low, const double &thre_high,
                       const bool &flag_in);
// radius filter
void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd_cloud,
                         const double &radius, const int &thre_count);
// convert to grid map data and publish
void SetMapTopicMsg(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    nav_msgs::OccupancyGrid &msg);

int main(int argc, char **argv) {
  ros::init(argc, argv, "pcl_filters");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  ros::Rate loop_rate(1.0);

  private_nh.param("file_directory", file_directory, std::string("/home/"));

  private_nh.param("file_name", file_name, std::string("map"));

  pcd_file = file_directory + file_name + pcd_format;

  private_nh.param("thre_z_min", thre_z_min, 0.2);
  private_nh.param("thre_z_max", thre_z_max, 2.0);
  private_nh.param("robot_reach_min", robot_reach_min, 0.1);
  private_nh.param("robot_reach_max", robot_reach_max, 0.8);
  private_nh.param("min_obstacle_height", min_obstacle_height, 0.3);
  private_nh.param("flag_pass_through", flag_pass_through, 0);
  private_nh.param("thre_radius", thre_radius, 0.1);
  private_nh.param("thres_point_count", thres_point_count, 10);
  private_nh.param("map_resolution", map_resolution, 0.05);
  private_nh.param("map_topic_name", map_topic_name, std::string("map"));

  ros::Publisher map_topic_pub =
      nh.advertise<nav_msgs::OccupancyGrid>(map_topic_name, 1);

  // load .pcd file
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file, *pcd_cloud) == -1) {
    PCL_ERROR("Couldn't read file: %s \n", pcd_file.c_str());
    return (-1);
  }

  std::cout << "Initial pointcloud number : " << pcd_cloud->points.size() << std::endl;

  // pass through filter - use the flag from launch parameter
  PassThroughFilter(thre_z_min, thre_z_max, (flag_pass_through != 0));
  // radius filter
  RadiusOutlierFilter(cloud_after_PassThrough_x, thre_radius, thres_point_count);
  // convert to grid map data and publish
  SetMapTopicMsg(cloud_after_Radius, map_topic_msg);

  while (ros::ok()) {
    map_topic_pub.publish(map_topic_msg);

    loop_rate.sleep();

    ros::spinOnce();
  }

  return 0;
}

// filter pointcloud using pass through
void PassThroughFilter(const double &thre_low, const double &thre_high,
                       const bool &flag_in) {

  // create filter_z
  pcl::PassThrough<pcl::PointXYZ> passthrough_z;
  // input pointcloud
  passthrough_z.setInputCloud(pcd_cloud);
  // set operation in z axis
  passthrough_z.setFilterFieldName("z");
  // set height range
  passthrough_z.setFilterLimits(thre_low, thre_high);
  // true : keep points out of range / false : keep points in the range
  passthrough_z.setFilterLimitsNegative(flag_in);
  // do filtering and save
  passthrough_z.filter(*cloud_after_PassThrough_z);

  // create filter_y
  pcl::PassThrough<pcl::PointXYZ> passthrough_y;
  // input pointcloud
  passthrough_y.setInputCloud(cloud_after_PassThrough_z);
  // set operation in y axis
  passthrough_y.setFilterFieldName("y");
  // set height range
  passthrough_y.setFilterLimits(-1000.0, 1000.0);
  // true : keep points out of range / false : keep points in the range
  passthrough_y.setFilterLimitsNegative(false);
  // do filtering and save
  passthrough_y.filter(*cloud_after_PassThrough_y);

  // create filter_x
  pcl::PassThrough<pcl::PointXYZ> passthrough_x;
  // input pointcloud
  passthrough_x.setInputCloud(cloud_after_PassThrough_y);
  // set operation in x axis
  passthrough_x.setFilterFieldName("x");
  // set height range
  passthrough_x.setFilterLimits(-1000.0, 1000.0);
  // true : keep points out of range / false : keep points in the range
  passthrough_x.setFilterLimitsNegative(false);
  // do filtering and save
  passthrough_x.filter(*cloud_after_PassThrough_x);

  // save to pcd file
  pcl::io::savePCDFile<pcl::PointXYZ>(file_directory + "map_filter.pcd",
                                      *cloud_after_PassThrough_x);
  std::cout << "pass through filter pointcloud : "
            << cloud_after_PassThrough_x->points.size() << std::endl;
}

// radius filter
void RadiusOutlierFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr &pcd_cloud0,
                         const double &radius, const int &thre_count) {
  // create filter
  pcl::RadiusOutlierRemoval<pcl::PointXYZ> radiusoutlier;
  // define input pointcloud
  radiusoutlier.setInputCloud(pcd_cloud0);
  // set radius and find point in range
  radiusoutlier.setRadiusSearch(radius);
  // delete points if < threshold
  radiusoutlier.setMinNeighborsInRadius(thre_count);
  radiusoutlier.filter(*cloud_after_Radius);
  // save to pcd file
  pcl::io::savePCDFile<pcl::PointXYZ>(file_directory + "map_radius_filter.pcd",
                                      *cloud_after_Radius);
  std::cout << "radius filter pointcloud : " << cloud_after_Radius->points.size()
            << std::endl;
}

// convert to grid map data and publish
void SetMapTopicMsg(const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    nav_msgs::OccupancyGrid &msg) {
  msg.header.seq = 0;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = "map";

  msg.info.map_load_time = ros::Time::now();
  msg.info.resolution = map_resolution;

  double x_min, x_max, y_min, y_max;
  double z_max_grey_rate = 0.05;
  double z_min_grey_rate = 0.95;

  double k_line =
      (z_max_grey_rate - z_min_grey_rate) / (thre_z_max - thre_z_min);
  double b_line =
      (thre_z_max * z_min_grey_rate - thre_z_min * z_max_grey_rate) /
      (thre_z_max - thre_z_min);

  if (cloud->points.empty()) {
    ROS_WARN("pcd is empty!\n");
    return;
  }

  for (int i = 0; i < cloud->points.size() - 1; i++) {
    if (i == 0) {
      x_min = x_max = cloud->points[i].x;
      y_min = y_max = cloud->points[i].y;
    }

    double x = cloud->points[i].x;
    double y = cloud->points[i].y;

    if (x < x_min)
      x_min = x;
    if (x > x_max)
      x_max = x;

    if (y < y_min)
      y_min = y;
    if (y > y_max)
      y_max = y;
  }
  // define origin position
  msg.info.origin.position.x = x_min;
  msg.info.origin.position.y = y_min;
  msg.info.origin.position.z = 0.0;
  msg.info.origin.orientation.x = 0.0;
  msg.info.origin.orientation.y = 0.0;
  msg.info.origin.orientation.z = 0.0;
  msg.info.origin.orientation.w = 1.0;
  // define grid map size
  msg.info.width = int((x_max - x_min) / map_resolution) + 1;
  msg.info.height = int((y_max - y_min) / map_resolution) + 1;
  // point coord (x,y) in real map corresponding to grid map coord [x*map.info.width+y]
  msg.data.resize(msg.info.width * msg.info.height);
  msg.data.assign(msg.info.width * msg.info.height, -1); // Initialize as unknown (-1)

  ROS_INFO("data size = %d\n", (int)msg.data.size());

  // Pass 1: Find the raw lowest point (potential floor) for each cell
  std::vector<float> min_z_raw(msg.data.size(), 10000.0f);
  for (int iter = 0; iter < cloud->points.size(); iter++) {
    int i = int((cloud->points[iter].x - x_min) / map_resolution);
    int j = int((cloud->points[iter].y - y_min) / map_resolution);
    if (i < 0 || i >= msg.info.width || j < 0 || j >= msg.info.height) continue;
    int idx = i + j * msg.info.width;
    float z = cloud->points[iter].z;
    if (z < min_z_raw[idx]) min_z_raw[idx] = z;
  }

  // Pass 2: Calibrate ground using a large separable min-filter (Radius ~1.0m)
  // This helps "borrow" the floor from up to 1 meter away to fill in missing scan data
  std::vector<float> temp_min_z(msg.data.size(), 10000.0f);
  int radius = 10; // 10 cells * 0.05m = 0.5m radius (1.0m diameter)

  // 2.1 Horizontal search
  for (int j = 0; j < msg.info.height; j++) {
    for (int i = 0; i < msg.info.width; i++) {
        int idx = i + j * msg.info.width;
        float self_z = min_z_raw[idx];
        float min_val = self_z;
        for (int di = -radius; di <= radius; di++) {
            int ni = i + di;
            if (ni >= 0 && ni < msg.info.width) {
                float v = min_z_raw[ni + j * msg.info.width];
                // Borrow ground only if it's not a huge jump (prevent Floor 1 "stealing" Floor 2 ground)
                if (v < min_val) {
                    if (self_z > 5000.0f || (self_z - v < 2.0f)) {
                        min_val = v;
                    }
                }
            }
        }
        temp_min_z[idx] = min_val;
    }
  }

  // 2.2 Vertical search
  std::vector<float> min_z_calibrated(msg.data.size(), 10000.0f);
  for (int i = 0; i < msg.info.width; i++) {
    for (int j = 0; j < msg.info.height; j++) {
        int idx = i + j * msg.info.width;
        float self_z = temp_min_z[idx];
        float min_val = self_z;
        for (int dj = -radius; dj <= radius; dj++) {
            int nj = j + dj;
            if (nj >= 0 && nj < msg.info.height) {
                float v = temp_min_z[i + nj * msg.info.width];
                // Borrow ground only if it's not a huge jump
                if (v < min_val) {
                    if (self_z > 5000.0f || (self_z - v < 2.0f)) {
                        min_val = v;
                    }
                }
            }
        }
        min_z_calibrated[idx] = min_val;
    }
  }

  // Pass 3: Collect thickness data within the robot's collision window
  std::vector<float> window_min_z(msg.data.size(), 10000.0f);
  std::vector<float> window_max_z(msg.data.size(), -10000.0f);
  for (int iter = 0; iter < cloud->points.size(); iter++) {
    int i = int((cloud->points[iter].x - x_min) / map_resolution);
    int j = int((cloud->points[iter].y - y_min) / map_resolution);
    if (i < 0 || i >= msg.info.width || j < 0 || j >= msg.info.height) continue;

    int idx = i + j * msg.info.width;
    float z = cloud->points[iter].z;
    float ground_z = min_z_calibrated[idx];

    // Check if point falls within the reach window above calibrated ground
    if (z > (ground_z + robot_reach_min) && z < (ground_z + robot_reach_max)) {
        if (z < window_min_z[idx]) window_min_z[idx] = z;
        if (z > window_max_z[idx]) window_max_z[idx] = z;
    }
  }

  // Pass 4: Finalize occupancy using the thickness threshold
  for (int idx = 0; idx < msg.data.size(); idx++) {
    if (window_max_z[idx] > -5000.0f) {
        float thickness = window_max_z[idx] - window_min_z[idx];
        if (thickness > min_obstacle_height) {
            msg.data[idx] = 100; // Confirmed obstacle
        } else {
            msg.data[idx] = 0;   // Noise or small bump -> Free
        }
    } else if (min_z_calibrated[idx] < 5000.0f) {
        msg.data[idx] = 0; // Clear floor -> Free
    }
  }
}

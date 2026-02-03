// SPDX-License-Identifier: BSD-2-Clause

#include <memory>
#include <iostream>

#include <boost/optional.hpp>

#include <ros/ros.h>
#include <ros/time.h>
#include <pcl_ros/point_cloud.h>

#include <std_msgs/Time.h>
#include <sensor_msgs/PointCloud2.h>
#include <hdl_graph_slam/FloorCoeffs.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/search.hpp>
#include <pcl/filters/impl/plane_clipper3D.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>

namespace hdl_graph_slam {

class FloorDetectionNodelet : public nodelet::Nodelet {
public:
  typedef pcl::PointXYZI PointT;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FloorDetectionNodelet() {}
  virtual ~FloorDetectionNodelet() {}

  virtual void onInit() {
    NODELET_DEBUG("initializing floor_detection_nodelet...");
    nh = getNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    points_sub = nh.subscribe("/filtered_points", 256, &FloorDetectionNodelet::cloud_callback, this);
    floor_pub = nh.advertise<hdl_graph_slam::FloorCoeffs>("/floor_detection/floor_coeffs", 32);

    read_until_pub = nh.advertise<std_msgs::Header>("/floor_detection/read_until", 32);
    floor_filtered_pub = nh.advertise<sensor_msgs::PointCloud2>("/floor_detection/floor_filtered_points", 32);
    floor_points_pub = nh.advertise<sensor_msgs::PointCloud2>("/floor_detection/floor_points", 32);
  }

private:
  /**
   * @brief initialize parameters
   */
  void initialize_params() {
    tilt_deg = private_nh.param<double>("tilt_deg", 0.0);                           // approximate sensor tilt angle [deg]
    sensor_height = private_nh.param<double>("sensor_height", 2.0);                 // approximate sensor height [m]
    height_clip_range = private_nh.param<double>("height_clip_range", 1.0);         // points with heights in [sensor_height - height_clip_range, sensor_height + height_clip_range] will be used for floor detection
    floor_pts_thresh = private_nh.param<int>("floor_pts_thresh", 512);              // minimum number of support points of RANSAC to accept a detected floor plane
    floor_normal_thresh = private_nh.param<double>("floor_normal_thresh", 10.0);    // verticality check thresold for the detected floor plane [deg]
    use_normal_filtering = private_nh.param<bool>("use_normal_filtering", true);    // if true, points with "non-"vertical normals will be filtered before RANSAC
    normal_filter_thresh = private_nh.param<double>("normal_filter_thresh", 20.0);  // "non-"verticality check threshold [deg]

    points_topic = private_nh.param<std::string>("points_topic", "/velodyne_points");
  }

  /**
   * @brief callback for point clouds
   * @param cloud_msg  point cloud msg
   */
  void cloud_callback(const sensor_msgs::PointCloud2ConstPtr& cloud_msg) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*cloud_msg, *cloud);

    if(cloud->empty()) {
      return;
    }

    // floor detection
    boost::optional<Eigen::Vector4f> floor = detect(cloud);

    // publish the detected floor coefficients
    hdl_graph_slam::FloorCoeffs coeffs;
    coeffs.header = cloud_msg->header;
    if(floor) {
      coeffs.coeffs.resize(4);
      for(int i = 0; i < 4; i++) {
        coeffs.coeffs[i] = (*floor)[i];
      }
    }

    floor_pub.publish(coeffs);

    // for offline estimation
    std_msgs::HeaderPtr read_until(new std_msgs::Header());
    read_until->frame_id = points_topic;
    read_until->stamp = cloud_msg->header.stamp + ros::Duration(1, 0);
    read_until_pub.publish(read_until);

    read_until->frame_id = "/filtered_points";
    read_until_pub.publish(read_until);
  }

  /**
   * @brief detect the floor plane from a point cloud
   * @param cloud  input cloud
   * @return detected floor plane coefficients
   * 
   * 处理流程说明:
   * 1. 补偿传感器倾斜角度
   * 2. 高度和法向量过滤预处理
   * 3. RANSAC平面拟合
   * 4. 垂直性验证
   * 5. 返回地板平面系数
   */
  boost::optional<Eigen::Vector4f> detect(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    // ===== 步骤1: 补偿传感器倾斜角度 =====
    // 构建倾斜补偿矩阵，用于校正传感器的倾斜安装角度
    // 绕Y轴旋转 tilt_deg 度，将倾斜的传感器坐标系对齐到水平坐标系
    Eigen::Matrix4f tilt_matrix = Eigen::Matrix4f::Identity();
    tilt_matrix.topLeftCorner(3, 3) = Eigen::AngleAxisf(tilt_deg * M_PI / 180.0f, Eigen::Vector3f::UnitY()).toRotationMatrix();

    // ===== 步骤2: RANSAC前的点云预过滤 =====
    // 目的：减少参与RANSAC计算的点数，提高效率和准确性
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    
    // 2.1 将点云转换到倾斜补偿后的坐标系
    pcl::transformPointCloud(*cloud, *filtered, tilt_matrix);
    
    // 2.2 高度裁剪：保留高度在 [sensor_height - height_clip_range, sensor_height + height_clip_range] 范围内的点
    // 第一次裁剪：移除高于 sensor_height + height_clip_range 的点（negative=false表示保留平面下方的点）
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height + height_clip_range), false);
    // 第二次裁剪：移除低于 sensor_height - height_clip_range 的点（negative=true表示保留平面上方的点）
    filtered = plane_clip(filtered, Eigen::Vector4f(0.0f, 0.0f, 1.0f, sensor_height - height_clip_range), true);

    // 2.3 法向量过滤：如果启用，则过滤掉法向量不垂直的点
    // 这一步进一步移除非地板点（如墙壁、倾斜表面等）
    if(use_normal_filtering) {
      filtered = normal_filtering(filtered);
    }

    // 2.4 将过滤后的点云变换回原始坐标系
    // 这样后续的RANSAC拟合结果是在原始坐标系下的
    pcl::transformPointCloud(*filtered, *filtered, static_cast<Eigen::Matrix4f>(tilt_matrix.inverse()));

    // 2.5 可选：发布过滤后的点云用于可视化和调试
    if(floor_filtered_pub.getNumSubscribers()) {
      filtered->header = cloud->header;
      floor_filtered_pub.publish(*filtered);
    }

    // ===== 步骤3: 点数检查 =====
    // 如果过滤后的点数太少，无法进行可靠的RANSAC拟合
    if(filtered->size() < floor_pts_thresh) {
      return boost::none;
    }

    // ===== 步骤4: RANSAC平面拟合 =====
    // 使用RANSAC算法从过滤后的点云中拟合平面模型
    pcl::SampleConsensusModelPlane<PointT>::Ptr model_p(new pcl::SampleConsensusModelPlane<PointT>(filtered));
    pcl::RandomSampleConsensus<PointT> ransac(model_p);
    ransac.setDistanceThreshold(0.1);  // 设置点到平面的距离阈值为0.1米
    ransac.computeModel();  // 执行RANSAC拟合

    // 4.1 获取内点（符合平面模型的点）
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    ransac.getInliers(inliers->indices);

    // 4.2 内点数量检查：如果内点太少，说明没有找到可靠的地板平面
    if(inliers->indices.size() < floor_pts_thresh) {
      return boost::none;
    }

    // ===== 步骤5: 垂直性验证 =====
    // 检查检测到的平面法向量是否接近垂直方向（地板应该是水平的）
    
    // 5.1 计算参考垂直方向（考虑传感器倾斜）
    Eigen::Vector4f reference = tilt_matrix.inverse() * Eigen::Vector4f::UnitZ();

    // 5.2 获取RANSAC拟合得到的平面系数 [A, B, C, D]
    // 其中平面方程为: Ax + By + Cz + D = 0, [A,B,C]是法向量
    Eigen::VectorXf coeffs;
    ransac.getModelCoefficients(coeffs);

    // 5.3 计算法向量与垂直方向的点积
    double dot = coeffs.head<3>().dot(reference.head<3>());
    // 如果夹角大于 floor_normal_thresh 度，则认为平面不够垂直，不是地板
    if(std::abs(dot) < std::cos(floor_normal_thresh * M_PI / 180.0)) {
      // the normal is not vertical
      return boost::none;
    }

    // 5.4 统一法向量方向：确保法向量指向上方（Z方向为正）
    // 这样平面系数的符号是一致的，便于后续处理
    if(coeffs.head<3>().dot(Eigen::Vector3f::UnitZ()) < 0.0f) {
      coeffs *= -1.0f;
    }

    // ===== 步骤6: 可选发布内点点云 =====
    // 用于可视化实际被识别为地板的点
    if(floor_points_pub.getNumSubscribers()) {
      pcl::PointCloud<PointT>::Ptr inlier_cloud(new pcl::PointCloud<PointT>);
      pcl::ExtractIndices<PointT> extract;
      extract.setInputCloud(filtered);
      extract.setIndices(inliers);
      extract.filter(*inlier_cloud);
      inlier_cloud->header = cloud->header;

      floor_points_pub.publish(*inlier_cloud);
    }

    // ===== 步骤7: 返回结果 =====
    // 返回地板平面系数 [A, B, C, D]，表示平面方程 Ax + By + Cz + D = 0
    return Eigen::Vector4f(coeffs);
  }

  /**
   * @brief plane_clip 使用平面裁剪点云
   * @param src_cloud 输入点云
   * @param plane 裁剪平面，格式为 [A, B, C, D]，表示平面方程 Ax + By + Cz + D = 0
   * @param negative 裁剪方向标志
   *                 - false: 保留平面下方/后方的点（法向量指向的反方向）
   *                 - true:  保留平面上方/前方的点（法向量指向的方向）
   * @return 裁剪后的点云
   * 
   * 工作原理：
   * 平面方程 Ax + By + Cz + D = 0 将空间分为两部分：
   * - Ax + By + Cz + D > 0 的点在平面的正侧（法向量[A,B,C]指向的一侧）
   * - Ax + By + Cz + D < 0 的点在平面的负侧（法向量相反方向）
   */
  pcl::PointCloud<PointT>::Ptr plane_clip(const pcl::PointCloud<PointT>::Ptr& src_cloud, const Eigen::Vector4f& plane, bool negative) const {
    // ===== 步骤1: 创建平面裁剪器 =====
    // PlaneClipper3D 根据平面方程对点云进行空间分割
    // 它会找出所有在平面某一侧的点的索引
    pcl::PlaneClipper3D<PointT> clipper(plane);
    pcl::PointIndices::Ptr indices(new pcl::PointIndices);

    // ===== 步骤2: 执行平面裁剪 =====
    // 找出所有满足 Ax + By + Cz + D < 0 的点的索引
    // 即平面法向量反方向（负侧）的点
    clipper.clipPointCloud3D(*src_cloud, indices->indices);

    // ===== 步骤3: 准备输出点云 =====
    pcl::PointCloud<PointT>::Ptr dst_cloud(new pcl::PointCloud<PointT>);

    // ===== 步骤4: 根据索引提取点 =====
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(src_cloud);    // 设置输入点云
    extract.setIndices(indices);          // 设置要提取的点的索引
    
    // 4.1 设置提取模式
    // negative=false: 提取索引中的点（平面负侧的点）
    // negative=true:  提取索引之外的点（平面正侧的点，即取反）
    extract.setNegative(negative);
    
    // 4.2 执行提取操作
    extract.filter(*dst_cloud);

    return dst_cloud;
  }

  /**
   * @brief filter points with non-vertical normals
   * @param cloud  input cloud
   * @return filtered cloud
   * 
   * 处理流程说明:
   * 1. 计算每个点的法向量
   * 2. 检查法向量与垂直方向的夹角
   * 3. 保留法向量接近垂直的点（地板候选点）
   * 4. 过滤掉法向量倾斜的点（墙壁、倾斜表面等）
   */
  pcl::PointCloud<PointT>::Ptr normal_filtering(const pcl::PointCloud<PointT>::Ptr& cloud) const {
    // ===== 步骤1: 初始化法向量估计器 =====
    // 法向量估计用于计算每个点周围局部表面的方向
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    ne.setInputCloud(cloud);

    // 1.1 创建KD树用于快速邻域搜索
    // KD树可以高效地找到每个点的k近邻点
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    ne.setSearchMethod(tree);

    // ===== 步骤2: 配置法向量计算参数 =====
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    
    // 2.1 设置K近邻数量为10
    // 即使用每个点周围最近的10个点来估计局部表面法向量
    // K值越大，法向量越平滑但可能丢失细节；K值越小，对噪声更敏感
    ne.setKSearch(10);
    
    // 2.2 设置视点位置为 (0, 0, sensor_height)
    // 视点用于确定法向量的方向一致性，使法向量朝向传感器
    // 这确保地板的法向量指向上方（朝向传感器）
    ne.setViewPoint(0.0f, 0.0f, sensor_height);
    
    // 2.3 执行法向量计算
    // 对点云中的每个点计算其局部表面法向量
    ne.compute(*normals);

    // ===== 步骤3: 准备输出点云 =====
    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>);
    filtered->reserve(cloud->size());  // 预分配内存以提高效率

    // ===== 步骤4: 基于法向量垂直性过滤点 =====
    // 遍历每个点，检查其法向量是否接近垂直方向
    for(int i = 0; i < cloud->size(); i++) {
      // 4.1 计算法向量与Z轴（垂直向上）的点积
      // 点积 = |normal| * |Z| * cos(θ)，由于都是单位向量，点积 = cos(θ)
      // θ 是法向量与垂直方向的夹角
      float dot = normals->at(i).getNormalVector3fMap().normalized().dot(Eigen::Vector3f::UnitZ());
      
      // 4.2 垂直性判断
      // 如果 |dot| > cos(normal_filter_thresh)，说明夹角 θ < normal_filter_thresh
      // 即法向量接近垂直方向（θ 较小），这样的点可能属于地板
      // 使用绝对值是因为法向量可能朝上或朝下，两个方向都认为是垂直的
      if(std::abs(dot) > std::cos(normal_filter_thresh * M_PI / 180.0)) {
        filtered->push_back(cloud->at(i));  // 保留垂直表面的点
      }
      // 法向量倾斜的点（如墙壁、斜坡）会被过滤掉
    }

    // ===== 步骤5: 设置输出点云属性 =====
    filtered->width = filtered->size();   // 点云宽度（无序点云设为点数）
    filtered->height = 1;                 // 点云高度（无序点云设为1）
    filtered->is_dense = false;           // 可能包含无效点（NaN/Inf）

    return filtered;
  }

private:
  ros::NodeHandle nh;
  ros::NodeHandle private_nh;

  // ROS topics
  ros::Subscriber points_sub;

  ros::Publisher floor_pub;
  ros::Publisher floor_points_pub;
  ros::Publisher floor_filtered_pub;

  std::string points_topic;
  ros::Publisher read_until_pub;

  // floor detection parameters
  // see initialize_params() for the details
  double tilt_deg;
  double sensor_height;
  double height_clip_range;

  int floor_pts_thresh;
  double floor_normal_thresh;

  bool use_normal_filtering;
  double normal_filter_thresh;
};

}  // namespace hdl_graph_slam

PLUGINLIB_EXPORT_CLASS(hdl_graph_slam::FloorDetectionNodelet, nodelet::Nodelet)

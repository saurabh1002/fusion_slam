#include <open3d/geometry/Image.h>

#include <Eigen/Core>
#include <string>
#include <vector>

#include "open3d/Open3D.h"
#include "yaml-cpp/yaml.h"

namespace o3d = open3d;

namespace datasets {
class freiburg1_desk {
public:
    using Image = open3d::geometry::Image;

    explicit freiburg1_desk(const std::string& data_root_dir,
                            const YAML::Node& cfg,
                            int n_scans = -1);

    [[nodiscard]] std::tuple<double, Eigen::Vector<double, 7>, Image, Image>
    operator[](int idx) const;
    [[nodiscard]] std::size_t size() const { return time_.size(); }

private:
    YAML::Node cfg_;

public:
    std::vector<double> time_;
    std::vector<Eigen::Vector<double, 7>> poses_;
    std::vector<std::string> rgb_files_;
    std::vector<std::string> depth_files_;
};

class freiburg1_desk_t {
public:
    using Image_t = open3d::t::geometry::Image;

    explicit freiburg1_desk_t(const std::string& data_root_dir,
                              const YAML::Node& cfg,
                              int n_scans = -1);

    [[nodiscard]] std::tuple<double, Eigen::Vector<double, 7>, Image_t, Image_t>
    operator[](int idx) const;
    [[nodiscard]] std::size_t size() const { return time_.size(); }

private:
    YAML::Node cfg_;

public:
    std::vector<double> time_;
    std::vector<Eigen::Vector<double, 7>> poses_;
    std::vector<std::string> rgb_files_;
    std::vector<std::string> depth_files_;
};

class maskrcnn {
public:
    explicit maskrcnn(freiburg1_desk data_fr1);
    [[nodiscard]] std::tuple<std::vector<std::string>,
                             std::vector<std::vector<int>>,
                             std::vector<o3d::geometry::Image>>
    operator[](int idx) const;
    [[nodiscard]] std::size_t size() const { return data_fr1_.time_.size(); }

private:
    freiburg1_desk data_fr1_;
};

}  // namespace datasets
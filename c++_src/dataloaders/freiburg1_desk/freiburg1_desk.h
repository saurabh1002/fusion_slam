#include <open3d/geometry/Image.h>

#include <Eigen/Core>
#include <string>
#include <vector>

#include "open3d/Open3D.h"
#include "yaml-cpp/yaml.h"

namespace o3d = open3d;

namespace datasets {
class freiburg1_desk {
private:
    using Image = o3d::geometry::Image;
    using Image_t = o3d::t::geometry::Image;

public:
    explicit freiburg1_desk(const std::string& data_root_dir,
                            const YAML::Node& cfg,
                            int n_scans = -1);

    [[nodiscard]] std::tuple<double, Eigen::Vector<double, 7>, Image, Image> At(
            int idx) const;
    [[nodiscard]] std::tuple<double, Eigen::Vector<double, 7>, Image_t, Image_t>
    At_t(int idx) const;
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
private:
    using Image = o3d::geometry::Image;
    using Image_t = o3d::t::geometry::Image;

public:
    explicit maskrcnn(freiburg1_desk data_fr1);
    [[nodiscard]] std::tuple<std::vector<std::string>,
                             std::vector<float>,
                             std::vector<std::vector<int>>,
                             std::vector<Image>>
    At(int idx) const;

    [[nodiscard]] std::tuple<std::vector<std::string>,
                             std::vector<float>,
                             std::vector<std::vector<int>>,
                             std::vector<Image_t>>
    At_t(int idx) const;

    [[nodiscard]] std::size_t size() const { return data_fr1_.time_.size(); }

private:
    freiburg1_desk data_fr1_;
};

}  // namespace datasets
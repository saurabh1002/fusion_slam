#include <open3d/geometry/Image.h>

#include <Eigen/Core>
#include <string>
#include <vector>

#include "open3d/Open3D.h"
#include "yaml-cpp/yaml.h"

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
    std::vector<double> time_;
    std::vector<Eigen::Vector<double, 7>> poses_;
    std::vector<std::string> rgb_files_;
    std::vector<std::string> depth_files_;
};
}  // namespace datasets
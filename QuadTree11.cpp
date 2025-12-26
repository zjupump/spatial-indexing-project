#include "QuadTree.h"
#include <set>
#include <limits>  // 添加这个头文件

namespace hw6 {

    /*
     * QuadNode
     */
void QuadNode::split(size_t capacity) {
    for (int i = 0; i < 4; ++i) {
        delete children[i];
        children[i] = nullptr;
    }

    // 计算四个子象限的边界框
    double midX = (bbox.getMinX() + bbox.getMaxX()) / 2.0;
    double midY = (bbox.getMinY() + bbox.getMaxY()) / 2.0;

    // 左下、右下、左上、右上四个子象限
    Envelope swBox(bbox.getMinX(), midX, bbox.getMinY(), midY);  // SW (southwest)
    Envelope seBox(midX, bbox.getMaxX(), bbox.getMinY(), midY);  // SE (southeast)
    Envelope nwBox(bbox.getMinX(), midX, midY, bbox.getMaxY());  // NW (northwest)
    Envelope neBox(midX, bbox.getMaxX(), midY, bbox.getMaxY());  // NE (northeast)

    // 创建四个子节点
    children[0] = new QuadNode(swBox);  // SW
    children[1] = new QuadNode(seBox);  // SE
    children[2] = new QuadNode(nwBox);  // NW
    children[3] = new QuadNode(neBox);  // NE

    // 将当前节点的特征分配到子节点中
    for (const Feature& feature : features) {
        const Envelope& featureBox = feature.geometry->getEnvelope();
        
        // 检查特征的边界框是否与每个子象限相交，如果相交则添加到对应子节点
        if (swBox.intersect(featureBox))
            children[0]->add(feature);
        if (seBox.intersect(featureBox))
            children[1]->add(feature);
        if (nwBox.intersect(featureBox))
            children[2]->add(feature);
        if (neBox.intersect(featureBox))
            children[3]->add(feature);
    }

    // 清空当前节点的特征列表
    features.clear();

    // 递归分裂子节点，如果子节点的特征数量超过容量限制
    for (int i = 0; i < 4; ++i) {
        if (children[i]->getFeatureNum() > capacity) {
            children[i]->split(capacity);
        }
    }
}

    void QuadNode::countNode(int& interiorNum, int& leafNum) {
        if (isLeafNode()) {
            ++leafNum;
        }
        else {
            ++interiorNum;
            for (int i = 0; i < 4; ++i)
                children[i]->countNode(interiorNum, leafNum);
        }
    }

    int QuadNode::countHeight(int height) {
        ++height;
        if (!isLeafNode()) {
            int cur = height;
            for (int i = 0; i < 4; ++i) {
                height = std::max(height, children[i]->countHeight(cur));
            }
        }
        return height;
    }

void QuadNode::rangeQuery(const Envelope& rect, std::vector<Feature>& result) {
    if (!bbox.intersect(rect))
        return;

    // 如果是叶节点，检查当前节点的所有特征
    if (isLeafNode()) {
        for (size_t i = 0; i < features.size(); ++i) {
            const Envelope& featureBox = features[i].geometry->getEnvelope();
            // Filter step: 检查特征的边界框是否与查询区域相交
            if (featureBox.intersect(rect)) {
                // Refinement step: 精确判断几何特征是否与查询区域相交
                if (features[i].geometry->intersect(rect)) {
                    result.push_back(features[i]);  // 添加到结果向量，而不是原特征向量
                }
            }
        }
    } 
    // 如果是内部节点，递归查询子节点
    else {
        for (int i = 0; i < 4; ++i) {
            children[i]->rangeQuery(rect, result);
        }
    }
}

void QuadTree::rangeQuery(const Envelope& rect, std::vector<Feature>& features) {
    features.clear();

    // 只有当根节点存在且与查询区域相交时才进行查询
    if (root) {
        std::vector<Feature> candidates;
        root->rangeQuery(rect, candidates);
        
        // Filter and refinement steps are already handled in QuadNode::rangeQuery
        // Here we just copy the results
        features = candidates;
    }
}

QuadNode* QuadNode::pointInLeafNode(double x, double y) {
    // 检查点是否在当前节点的边界框内
    if (!bbox.contain(x, y)) {
        return nullptr;
    }

    // 如果是叶节点，直接返回当前节点
    if (isLeafNode()) {
        return this;
    }

    // 如果是内部节点，递归搜索包含点的子节点
    double midX = (bbox.getMinX() + bbox.getMaxX()) / 2.0;
    double midY = (bbox.getMinY() + bbox.getMaxY()) / 2.0;

    // 判断点在哪个象限
    if (x < midX) {
        if (y < midY) {
            // SW象限
            return children[0]->pointInLeafNode(x, y);
        } else {
            // NW象限
            return children[2]->pointInLeafNode(x, y);
        }
    } else {
        if (y < midY) {
            // SE象限
            return children[1]->pointInLeafNode(x, y);
        } else {
            // NE象限
            return children[3]->pointInLeafNode(x, y);
        }
    }
}

bool QuadTree::NNQuery(double x, double y, std::vector<Feature>& features) {
    if (!root || !(root->getEnvelope().contain(x, y))) {
        return false;
    }

    // 通过pointInLeafNode查询点(x, y)所在的叶节点
    QuadNode* leafNode = root->pointInLeafNode(x, y);
    if (!leafNode) {
        return false;
    }

    // 计算查询点(x, y)与该叶节点内的几何特征包围盒的最大距离的最小值minDist
    double minDist = std::numeric_limits<double>::max();
    bool hasFeatureInLeaf = false;

    for (size_t i = 0; i < leafNode->getFeatureNum(); ++i) {
        const Feature& feature = leafNode->getFeature(i);
        const Envelope& featureBox = feature.geometry->getEnvelope();
        
        // 使用maxDistance2Envelope函数，获得查询点到几何对象包围盒的最短的最大距离
        double dist = featureBox.maxDistance2Point(x, y);
        if (dist < minDist) {
            minDist = dist;
            hasFeatureInLeaf = true;
        }
    }

    // 如果叶节点无几何特征，使用叶节点边界框的尺寸作为初始距离
    if (!hasFeatureInLeaf) {
        minDist = std::max(leafNode->getEnvelope().getWidth(), 
                          leafNode->getEnvelope().getHeight());
    }

    // 构造查询区域 (x – minDist, x + minDist, y – minDist, y + minDist)
    Envelope queryRect(x - minDist, x + minDist, y - minDist, y + minDist);

    // 查询几何特征的包围盒与该区域相交的几何特征（filter）
    std::vector<Feature> candidates;
    root->rangeQuery(queryRect, candidates);

    // 在候选集中查询与查询点(x, y)距离最近的几何特征（refinement）
    double minActualDist = std::numeric_limits<double>::max();
    Feature nearestFeature;
    bool found = false;

    for (const Feature& feature : candidates) {
        double dist = feature.geometry->distance(Point(x, y));
        if (dist < minActualDist) {
            minActualDist = dist;
            nearestFeature = feature;
            found = true;
        }
    }

    if (found) {
        features.clear();
        features.push_back(nearestFeature);
        return true;
    }

    return false;
}
    void QuadNode::draw() {
        if (isLeafNode()) {
            bbox.draw();
        }
        else {
            for (int i = 0; i < 4; ++i)
                children[i]->draw();
        }
    }

    /*
     * QuadTree
     */
bool QuadTree::constructTree(const std::vector<Feature>& features) {
    if (features.empty())
        return false;

    // 计算所有特征的边界框
    if (features.size() > 0) {
        bbox = features[0].geometry->getEnvelope();
        for (size_t i = 1; i < features.size(); ++i) {
            bbox = bbox.unionEnvelope(features[i].geometry->getEnvelope());
        }
    } else {
        return false;
    }

    // 创建根节点
    root = new QuadNode(bbox);

    // 将所有特征添加到根节点
    for (const Feature& feature : features) {
        root->add(feature);
    }

    // 如果根节点特征数量超过容量，进行分裂
    if (root->getFeatureNum() > getCapacity()) {
        root->split(getCapacity());
    }

    return true;
}

    void QuadTree::countNode(int& interiorNum, int& leafNum) {
        interiorNum = 0;
        leafNum = 0;
        if (root)
            root->countNode(interiorNum, leafNum);
    }

    void QuadTree::countHeight(int& height) {
        height = 0;
        if (root)
            height = root->countHeight(0);
    }

void QuadTree::rangeQuery(const Envelope& rect, std::vector<Feature>& features) {
    features.clear();

    // 只有当根节点存在且与查询区域相交时才进行查询
    if (root && root->getEnvelope().intersect(rect)) {
        std::vector<Feature> candidates;
        root->rangeQuery(rect, candidates);
        
        // Filter step (选择查询区域与几何对象包围盒相交的几何对象)
        for (const Feature& feature : candidates) {
            const Envelope& featureBox = feature.geometry->getEnvelope();
            if (featureBox.intersect(rect)) {
                // Refinement step: 精确判断几何特征是否与查询区域相交
                if (feature.geometry->intersect(rect)) {
                    features.push_back(feature);
                }
            }
        }
    }

    // 注意四叉树区域查询仅返回候选集，精炼步在hw6的rangeQuery中完成
}

    // 删除重复的 NNQuery 函数实现

// 空间关联函数实现
    void spatialJoinWithIndex(const std::vector<Feature>& leftFeatures,
                              const std::vector<Feature>& rightFeatures,
                              double distanceThreshold,
                              std::vector<std::pair<Feature, Feature>>& result) {
        result.clear();
        
        if (leftFeatures.empty() || rightFeatures.empty()) {
            return;
        }

        // 为右侧特征集构建四叉树索引
        QuadTree spatialIndex(10);  // 容量设为10
        
        // 构建特征向量
        std::vector<Feature> rightFeatureVec;
        rightFeatureVec.reserve(rightFeatures.size());
        for (const auto& feature : rightFeatures) {
            rightFeatureVec.push_back(feature);
        }
        
        spatialIndex.constructTree(rightFeatureVec);

        // 对左侧特征集进行嵌套循环，但使用索引优化
        for (const Feature& leftFeature : leftFeatures) {
            const Envelope& leftEnv = leftFeature.geometry->getEnvelope();
            
            // 扩展查询边界框以包含距离阈值范围
            Envelope queryEnv(leftEnv.getMinX() - distanceThreshold,
                             leftEnv.getMaxX() + distanceThreshold,
                             leftEnv.getMinY() - distanceThreshold,
                             leftEnv.getMaxY() + distanceThreshold);

            // 使用四叉树索引查询候选特征
            std::vector<Feature> candidates;
            spatialIndex.rangeQuery(queryEnv, candidates);

            // 精确计算距离并筛选结果
            for (const Feature& rightFeature : candidates) {
                double actualDistance = leftFeature.geometry->distance(rightFeature.geometry);
                
                if (actualDistance <= distanceThreshold) {
                    result.push_back(std::make_pair(leftFeature, rightFeature));
                }
            }
        }
    }

    // 针对特定场景的便捷函数：查询道路一定距离内的公共自行车站点
    void findStationsNearRoads(const std::vector<Feature>& roads,
                              const std::vector<Feature>& stations,
                              double distanceThreshold,
                              std::vector<std::pair<Feature, Feature>>& result) {
        // 以道路为左特征集，站点为右特征集
        spatialJoinWithIndex(roads, stations, distanceThreshold, result);
    }
    
    // 反向查询：查询站点一定距离内的道路
    void findRoadsNearStations(const std::vector<Feature>& stations,
                              const std::vector<Feature>& roads,
                              double distanceThreshold,
                              std::vector<std::pair<Feature, Feature>>& result) {
        // 以站点为左特征集，道路为右特征集
        spatialJoinWithIndex(stations, roads, distanceThreshold, result);
    }
    
} // namespace hw6
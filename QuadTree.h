#ifndef QUADTREE_H_INCLUDED
#define QUADTREE_H_INCLUDED

#include "Geometry.h"
#include "Tree.h"
#include <string>
#include <vector>      // 添加这行
#include <utility>     // 添加这行，用于 std::pair

namespace hw6 {

    class QuadNode {
    private:
        Envelope bbox;
        QuadNode* children[4];
        std::vector<Feature> features;

    public:
        QuadNode() = delete;
        QuadNode(const Envelope& box) : bbox(box) {
            children[0] = children[1] = children[2] = children[3] = nullptr;
        }

        ~QuadNode() {
            for (int i = 0; i < 4; ++i) {
                delete children[i];
                children[i] = nullptr;
            }
        }

        bool isLeafNode() { return children[0] == nullptr; }

        const Envelope& getEnvelope() { return bbox; }

        QuadNode* getChildNode(size_t i) { return i < 4 ? children[i] : nullptr; }

        size_t getFeatureNum() const { return features.size(); }

        const Feature& getFeature(size_t i) const { return features[i]; }

        void add(const Feature& f) { features.push_back(f); }

        void add(const std::vector<Feature>& fs) {
            features.insert(features.begin(), fs.begin(), fs.end());
        }

        void countNode(int& interiorNum, int& leafNum);

        int countHeight(int height);

        void draw();

        // split the node into four child nodes, assign each feature to its
        // overlaped child node(s), clear feature vector, and split child node(s) if
        // its number of features is larger than capacity
        void split(size_t capacity);

        void rangeQuery(const Envelope& rect, std::vector<Feature>& features);

        QuadNode* pointInLeafNode(double x, double y);
    };

    class QuadTree : public Tree {
    private:
        QuadNode* root;

    public:
        QuadTree() : Tree(5), root(nullptr) {}
        QuadTree(size_t cap) : Tree(cap), root(nullptr) {}
        ~QuadTree() {
            if (root != nullptr)
                delete root;
            root = nullptr;
        }

        virtual bool constructTree(const std::vector<Feature>& features) override;

        virtual void countNode(int& interiorNum, int& leafNum) override;

        virtual void countHeight(int& height) override;

        virtual void rangeQuery(const Envelope& rect, std::vector<Feature>& features) override;

        virtual bool NNQuery(double x, double y, std::vector<Feature>& features) override;

        QuadNode* pointInLeafNode(double x, double y) {
            return root->pointInLeafNode(x, y);
        }

        virtual void draw() override;

    public:
        static void test(int t);

        static void analyse();
    };

    void spatialJoinWithIndex(const std::vector<Feature>& leftFeatures,
                              const std::vector<Feature>& rightFeatures,
                              double distanceThreshold,
                              std::vector<std::pair<Feature, Feature>>& result);

    void findStationsNearRoads(const std::vector<Feature>& roads,
                              const std::vector<Feature>& stations,
                              double distanceThreshold,
                              std::vector<std::pair<Feature, Feature>>& result);

    void findRoadsNearStations(const std::vector<Feature>& stations,
                              const std::vector<Feature>& roads,
                              double distanceThreshold,
                              std::vector<std::pair<Feature, Feature>>& result);

} // namespace hw6

#endif
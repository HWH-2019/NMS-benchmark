#include<vector>
#include<string>
#include<iostream>
#include<algorithm>

struct bbox
{
    int x1,y1,x2,y2;
    float score;
    bbox(int x1,int y1,int x2,int y2,float score)
    {
        this->x1=x1;
        this->y1=y1;
        this->x2=x2;
        this->y2=y2;
        this->score=score;
    }
    std::string __str__(){
        return "bbox(" + std::to_string(x1) + "," + std::to_string(y1) + "," + std::to_string(x2) + "," + std::to_string(y2) + "," + std::to_string(score) + ")";
    }
};

// 计算 iou
float iou(bbox bbox1, bbox bbox2){
    float area1 = (bbox1.x2 - bbox1.x1 + 1) * (bbox1.y2 - bbox1.y1 + 1);
    float area2 = (bbox2.x2 - bbox2.x1 + 1) * (bbox2.y2 - bbox2.y1 + 1);

    int x11 = std::max(bbox1.x1, bbox2.x1);
    int y11 = std::max(bbox1.y1, bbox2.y1);
    int x22 = std::max(bbox2.x2, bbox2.x2);
    int y22 = std::max(bbox2.y2, bbox2.y2);
    float intersection = (x22 - x11 + 1) * (y22 - y11 + 1);
    return intersection / (area1 + area2 - intersection);
}

std::vector<bbox> nms(std::vector<bbox> &bboxs, float threshold = 0.5){
    auto cmp = [](bbox bbox1, bbox bbox2){
        return bbox1.score < bbox2.score;
    };

    // sorted by score
    std::sort(bboxs.begin(), bboxs.end(), cmp);

    std::vector<bbox> res;
    while(!bboxs.empty()){
        res.emplace_back(bboxs.back());
        bboxs.pop_back();
        for(size_t i = 0; i < bboxs.size(); ++i){
            if(iou(bboxs[i], res.back()) > threshold){
                bboxs.erase(bboxs.begin() + i);
            }
        }
    }

    return res;
}


int main(){
    std::vector<bbox> bboxs;
    bboxs.emplace_back(bbox(1,1,2,2,0.77));
    bboxs.emplace_back(bbox(2,0,4,3,0.28));
    bboxs.emplace_back(bbox(0,0,1,1,0.34));
    bboxs.emplace_back(bbox(0,4,3,3,0.5));

    std::vector<bbox> res = nms(bboxs);
    for(auto r : res){
        std::cout<<r.__str__()<<std::endl;
    }
}


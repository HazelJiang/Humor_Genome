import React from "react";
import ReactDOM from "react-dom";
import "antd/dist/antd.css";
import "./index.css";
import { Tabs } from "antd";
import { Timeline } from "antd";

const TabPane = Tabs.TabPane;

function callback(key) {
  console.log(key);
}

ReactDOM.render(
  <div className="container">
  <Tabs defaultActiveKey="1" onChange={callback}>
    <TabPane tab="Naive Bayes" key="1">
    <Timeline>
        <Timeline.Item color="green">
        How this classifier works
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="green">
        How it finds the comedian
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="green">
        Our Results
        <p>p1</p>
        <p>p2</p>
        <p>p3</p>
        </Timeline.Item>
        <Timeline.Item color="green">
        Sources
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
      </Timeline>
    </TabPane>
    <TabPane tab="SVM" key="2">
    <Timeline>
        <Timeline.Item color="blue">
        How this classifier works
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="blue">
        How it finds the comedian
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="blue">
        Our Results
        <p>p1</p>
        <p>p2</p>
        <p>p3</p>
        </Timeline.Item>
        <Timeline.Item color="blue">
        Sources
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
      </Timeline>
    </TabPane>
    <TabPane tab="Decision Tree" key="3">
    <Timeline>
        <Timeline.Item color="pink">
        How this classifier works
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="pink">
        How it finds the comedian
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="pink">
        Our Results
        <p>p1</p>
        <p>p2</p>
        <p>p3</p>
        </Timeline.Item>
        <Timeline.Item color="pink">
        Sources
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
      </Timeline>
    </TabPane>
    <TabPane tab="Kth Nearest Neighbor" key="4">
    <Timeline>
        <Timeline.Item color="brown">
        How this classifier works
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="brown">
        How it finds the comedian
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
        <Timeline.Item color="brown">
        Our Results
        <p>p1</p>
        <p>p2</p>
        <p>p3</p>
        </Timeline.Item>
        <Timeline.Item color="brown">
        Sources
        <p>p1</p>
        <p>p2</p>
        </Timeline.Item>
      </Timeline>
    </TabPane>    
  </Tabs>
  </div>,
  document.getElementById("container")
);

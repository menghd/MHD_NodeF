# Souray 三项目统一工作约定

自`2026_09_09_10_30_34`起适用于LOOK、Radon_Bridge和MHD_Project的新准备版本。统一外层工作目录和溯源，不统一或改写三项目的科学算法。

## 布局

| 内容 | ws02 | Ibex |
|---|---|---|
| 代码根 | `/home/mengh` | `/ibex/project/c2377/souray/home/mengh` |
| 数据根 | `/data/mengh` | `/ibex/project/c2377/souray/data/mengh` |

- 代码隔离检出：`<代码根>/<项目>/<时间戳>/`，不加projects层。
- 本轮结果：`<数据根>/<项目>/runs/<时间戳>/`。
- 共享原始数据：`<数据根>/UKBiobank/ophthalmology/raw/`；清单在相邻manifests，派生缓存须注明项目/协议/预处理SHA。跨项目共享原始数据不等于共享划分或test访问授权。
- 环境：`<代码根>/environments/<项目小写>_<时间戳>/`，按依赖兼容性隔离，不复制跨机器虚拟环境。
- 系统HOME不修改；VS Code连接后打开对应代码路径。计算通过Slurm或工作站已批准调度执行。

例如Ibex的三个代码路径分别为：
`.../home/mengh/LOOK/2026_09_09_10_30_34/`、
`.../home/mengh/Radon_Bridge/2026_09_09_10_30_34/`、
`.../home/mengh/MHD_Project/2026_09_09_10_30_34/`。

LOOK的Git仓库内部保留home/data布局，因此这个完整检出中的可执行应用根是`<代码检出>/home/`；其仓库data只是受控模板，不是临床数据。本次不搬动正在运行的旧LOOK发布目录。Radon_Bridge与MHD_Project的应用根就是检出根。

## 命名与依赖

GitHub名称固定LOOK、Radon_Bridge、MHD_Project；R&B展示名称为Radon Bridge（R&B），其新Python包为radon_bridge。LOOK和MHD已有模块/API保留原语义，不机械替换类名。

MHD_Project是版本化框架依赖。应用的每个准备版本必须记录依赖提交和API代次，最新main不能作为可变运行依赖；现有V4研究不因仓库更新自动转为V5。共享目录的框架检出用于开发/参考，正式研究可保留固定子模块，禁止依赖他人可随时修改的共享源码。

## 元数据与执行

每个运行记录：项目/时间戳、代码提交、框架提交/API、环境版本、数据/划分/预处理摘要、协议、设备映射、启动/停止与验收状态。准备完成不等于训练或迁移验收通过。

本次只统一命名和运行准备；GPU申请暂停，新扩展实验未锁定。Slurm分配的CUDA_VISIBLE_DEVICES不可被工作站物理编号覆盖。任务结束即退出，不用假负载占卡。

历史报告、配置、数据、检查点及SHA不改写；历史发布路径保留。Radon_Bridge改名所需的旧路径/导入兼容仅限历史解析，不要求LOOK/MHD新算法保留额外别名。本约定不改变任何项目现有显存、test或监控规则。

## 使用

```bash
python workspace/paths.py --machine ibex --project Radon_Bridge --timestamp 2026_09_09_10_30_34
```

该工具只返回路径，不自动创建、安装或启动任务。`registry.json`及`paths.py`在三个仓库保持相同，修改时同步校验。根路径可通过独立registry文件配置，不写入数学模块。

## 同一套规范的维护

这套规范的主维护副本位于`MHD_Project/workspace/`，LOOK和Radon_Bridge携带逐字节一致的副本以便独立检出和离线使用。规范只负责工作区管理，不向MHD框架算法引入应用依赖。修改时在三个仓库的同一准备批次同步，通过`check.py --peer ...`核对，记录内容SHA；不得只在某台服务器改一份。

优先级：用户当前明确要求、当前项目已批准的研究/数据协议、共同管理规范、历史默认。统一管理不覆盖科学约束。LOOK/R&B运行时各用自己的参数与显存限制；项目间不共享同一可变Python环境或运行锁。

所有提交遵循隔离修改→必要测试→GitHub同步→新时间戳部署→文件校验→明确批准的任务启动。主分支是当前维护入口；时间戳准备分支保留同批变更，正式开始实验时另锁定不可变提交/源清单。准备分支还在验收时不得伪称正式训练快照。

旧路径兼容仅用于R&B此次用户批准的命名迁移；LOOK原有禁止新增兼容包装的标准继续适用于LOOK算法。代码层API升级应单列测试，不能仅靠统一文件名认定兼容。

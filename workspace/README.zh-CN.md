# 统一工作区规范

以 [REPOSITORY_STANDARD.md](REPOSITORY_STANDARD.md) 为准。LOOK 与 Radon_Bridge
共享研发目录结构、安装与检查命令；MHD_Project 独立作为可安装工具箱。

时间戳表示一轮研究，代码提交 SHA 表示实际版本。MHD 更新独立进行，研究项目
使用子模块提交、API 版本和文件 SHA 固定依赖，默认不追随最新版。
服务器活动区只保留当前轮次和必需依赖；历史代码及汇总材料上 GitHub，受限数据
与大检查点进授权归档，核验后才能清理工作副本。模板不自动删除或启动训练。

`python workspace/check.py --peer /path/to/another/repository` 校验共同文件。
`python scripts/manage.py check` 校验研发项目结构及固定框架。
主机路径由 registry.json 管理，新服务器只增加根路径，不改科学实现。

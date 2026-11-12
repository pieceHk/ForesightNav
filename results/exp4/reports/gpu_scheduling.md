# Exp4 三卡调度与资源调整记录

2026年11月11日，用户先授权使用服务器空闲GPU，随后明确要求释放GPU0，仅在GPU1、2、3运行。当前调度范围已固定为这三张A40，建图与导航在每个场景的同一卡上运行。四个变体仍按Full、Nearest、Myopic、Suffix-only依次评估，每组固定400条，不跨变体并行。

## 已发生的接续

时间均为北京时间。

| 时间 | 动作与结果 |
| --- | --- |
| 11月10日18:12 | 原Full控制器在GPU2、3启动 |
| 11月11日10:32:45 | 空闲卡扩容接续，GPU0启动cvZr5TUy5C5，GPU1启动mL8ThkuaVTM；原GPU2、3场景进程继续运行 |
| 11月11日10:34:00 | 按用户新要求释放GPU0，仅停止本实验在该卡的启动器及子进程。中止时完整原生回合数为0，初始化日志和原启动记录保留 |
| 11月11日10:36:14 | 三卡控制器接续，保留GPU1、2、3现有启动器与角色进程；cvZr5TUy5C5重新排入未执行场景队列，待三张授权卡之一空闲后完整运行20条 |

GPU0的中止由资源授权变化触发，不依据成功率或失败结果。被中止的初始化实例不进入正式400条汇总，也没有用其他episode替换其20条固定身份。恢复实例使用独立日志和运行标签，原记录保留。该初始化的原生启动、状态、源码和角色环境等记录已单独[归档](../logs/native/resource_interruptions/full-cvZr5TUy5C5-gpu0-initialization.tar.gz)，原始66598字节、压缩7388字节，逐文件压缩往返SHA全部一致；[归档清单](../meta/gpu0-initialization-log-archive.json)明确完整回合为0及排除范围。

## 接续方式与检查

本次只增加results/exp4/code下的CPU调度工具。161项已验收运行输入、冻结标定文件、原evaluate_sequence.py以及五个共用指标分析组件保持原SHA。每个新场景仍调用原command、run.py和对应变体配置，场景内目标顺序、源episode ID、seed=1及测地距离实现不变。

旧CPU控制器暂时停止调度，其现存导航启动器继续运行。新控制器按PID、启动时刻和boot ID接管观察，保留各启动器的真实父进程。原启动器结束后，从Linux保留的退出状态读取真实returncode，再结合原生completed状态和20条覆盖检查确认完成。只有该父进程的全部子进程已退出后，才回收停止的CPU父进程；不向保留的GPU场景进程发送停止信号。

初版扩容通过10项调度检查，三卡限定版本另通过14项检查，包括GPU授权、占用和显存筛选、固定身份完整覆盖、重复场景拒绝、PID复用拒绝，以及真实Linux玩具进程的接续验证。玩具验证确认：停止CPU父进程后子进程仍推进，退出码0和17均按真实值捕获。这24项属于调度检查，未计入已有的60项原生实现检查与219次关闭模式回放。

11月11日11:28的[三卡启动审计](../diagnostics/full/parallel-scene-launch-audits/20261111T032856434695Z.json)已核对12个场景的240条身份，包括GPU1新增场景及GPU3恢复实例。[环境索引](../diagnostics/environment-index/20261111T032856307066Z.json)复核20条冒烟和12个已启动Full场景的双角色Python、锁文件和包清单签名，并核对实际使用的GPU1、2、3型号与驱动。GPU0旧初始化属于单独保留的资源中止记录。

## 后续口径

11月11日11:28的[真实接续核查](../diagnostics/gpu-expansion/20261111T023614070319Z/native-exit-audits/20261111T032833711934Z.json)确认，原GPU3启动器668266在完成XB4GS9ShBRE的全部20条回合后正常退出。新控制器记录的wait status为0，与当时仍由停止父进程保留的Linux退出记录逐字节SHA一致，另由ps核对进程为Z状态、父PID为468645。原生completed状态和20条身份覆盖也通过检查。这是首个实际接管场景的退出证据，未向GPU工作进程发送信号。GPU3随后启动cvZr5TUy5C5的独立恢复实例，20条固定身份、seed、配置、源码及角色环境均已核对。

GPU分配按meta/evaluation-gpu-policy.json限制为1、2、3。早期三卡控制器只领取无计算进程的卡；下述共享版本保留38 GiB空闲门槛，并允许有限的已有计算占用。后续三个变体继续使用原CP4及前驱完整性门禁。Full完成后发布并汇报CP4，再启动Nearest。

规划耗时保留完整分布与尾值，并披露Full运行期间两卡、短暂扩容和三卡资源阶段。不同资源阶段的耗时变化不能单独归因于BLP机制。实际成功率及路径指标继续按完整400条原生结果计算，不筛除慢回合或失败回合。

11月11日12:15的[资源快照](../diagnostics/resource-context/20261111T041507092641Z.json)记录到新的外部共享任务：GPU1、2、3各有4个外部计算进程，合计分别占8032、8032、7596 MiB；主机384个逻辑核，load1=197.35，CPU some/full PSI avg10均为0。12:16跟进时本实验已完成211条，三卡仍在推进，源码、参数绑定和记录错误检查正常。该快照不证明共享任务对速度或成功率的因果影响；保留这段全部原生结果与规划时延。公开记录仅保留资源数量和占用，不包含外部用户名、PID、命令或环境；完整只读诊断保存在本机工作区，该次诊断未发送停止信号。

11月11日12:57:57，bxsVRursffK的20条固定回合全部完成，原启动器696782的Linux wait status由控制器捕获为0。该停止父进程保留的XB和bxs启动器均已正常退出，控制器随后按既定规则[回收旧CPU父进程468645](../diagnostics/gpu-expansion/20261111T023614070319Z/predecessor-retired-468645.json)，没有停止GPU worker。父进程回收后，旧/proc退出记录已不存在；本次补查核对的是控制器历史退出记录与原生20条完成状态，不将其表述为再次独立读取原/proc记录。

12:59:36的[队列与资源核查](../diagnostics/resource-context/20261111T045936720840Z-queue.json)显示GPU2已无本实验计算进程，但仍有4个外部进程、共8032 MiB，空闲36633 MiB。当时控制器要求整卡无计算进程且至少38912 MiB空闲，因此GPU2未领取下一场景。GPU1、3的原场景继续运行，8个尚未启动场景保持原顺序等待。此前[12:56核查](../diagnostics/resource-context/20261111T045634810671Z-queue.json)中三卡仍各有本实验的两个计算进程。两次只读核查均绑定实际调度源码、冻结参数及原生输入，没有改动启动门槛、停止其他任务或改变评估身份。

13:18的[CPU资源快照](../diagnostics/resource-context/20261111T051844309890Z-cpu-pressure.json)记录到新的CPU拥堵：384个逻辑核上load1为2215.6，CPU some PSI avg10为85.10%。最大外部用户在2.246秒采样中占用约366.6个逻辑核；采样中CPU最高的15个worker来自同一父进程，各有192–194个线程。该用户的全部进程包含多组任务，不将其所有线程都归入这一个父进程。公开快照省略外部用户名与PID。原导航任务和观察器仍存活，原生结果及全部规划时延继续保留；只读查询曾因观察快照超过120秒而结束，未据此重启原生任务。本机后续观察窗口把快照容忍度显式记录为300秒，仍核对进程身份、源码、参数、记录错误和实际轨迹，服务器原观察器保持原样。

外部GPU任务随后减少。13:24:57实读GPU2空闲41176 MiB、已有compute占用4016 MiB，足以满足原有38912 MiB空闲门槛。共享调度版本允许已有compute占用不超过8192 MiB，并继续限制为GPU1、2、3及每卡一个本实验场景。它保留原场景命令、固定20条身份、seed、原生源码和冻结参数，新增多代CPU父进程的出生记录核查。该版本的[23项调度检查和实际预检](../diagnostics/shared-gpu-scheduling-preflight/20261111T052134753952Z.json)通过，包括真实Linux玩具子进程的持续推进和0/17退出码读取；这些检查独立于60项原生实现检查及219次关闭模式回放。

13:24:58的[实际接续](../diagnostics/gpu-resource-sharing/20261111T052457548416Z/handoff.json)把CPU控制器更换为777321，原ML启动器703343、CV启动器724204及观察器468659继续运行，没有重启GPU场景。旧CPU704760停止保留CV的真实退出状态，703335继续保留ML，二者只在对应原生子进程退出后回收。GPU2领取mv2HUxq3B53的完整20条，[启动核查](../diagnostics/full/parallel-scene-launch-audits/20261111T053005561859Z.json)覆盖13个实际启动场景的260条身份，目标顺序、source ID、seed和运行源码一致。13:31的[启动观察](../diagnostics/gpu-resource-sharing/20261111T052457548416Z/native-startup-MV-20261111T053140173020Z.json)已见该场景19个真实导航动作；[资源核查](../diagnostics/resource-context/20261111T053138713900Z-shared-queue.json)确认三卡各有两个本实验compute进程，GPU2仍空闲20857 MiB。当前版本不停止外部任务，也不使用GPU0或4–7。

上述CPU拥堵、GPU2等待及共享接续均属于本次Full的运行条件，后续CP4/CP5应随过程指标披露。规划耗时与回退率的变化不能单独归因于BLP；所有成功、失败、预算耗尽回合和时延尾值均按完整固定任务池保留。

13:34的[环境索引](../diagnostics/environment-index/20261111T053402670639Z.json)已核对20条冒烟和13个已启动Full场景的mapping/navigation双角色记录，签名与验收环境一致，GPU分配和运行源码匹配。其余7个场景尚未启动，后续仍需刷新索引。

13:50的[CPU跟进采样](../diagnostics/resource-context/20261111T055039585799Z-cpu-pressure.json)中，load1为2110.11、CPU some PSI avg10为82.15%，最大外部用户采样占用约367.9个逻辑核。三卡导航继续运行，CPU拥堵仍未解除；本次跟进没有修改实验或停止其他任务。

14:00已核查GPU2新场景的[首条完整回合](../diagnostics/completed-prefix/full/mv2HUxq3B53/0.json)：463步，三阶段成功向量为0/1/0。其身份、冻结参数、三个阶段的轨迹和STOP记录均通过检查；导航失败的两个阶段照常保留。这项核查验证了共享接续后的记录一致性，不代表该场景20条或Full400已完成。

14:40以当前共享调度执行了[后续三个变体的只读启动预检](../diagnostics/shared-remaining-stage-preflight/20261111T064051521665Z.json)：Nearest因Full未完成而不可启动，Myopic和Suffix-only因各自前驱记录不存在而不可启动。三次CLI均返回ready=false，未新增后续阶段文件或worker，原生源码与参数不变。这项预检核对当前的前驱拒绝行为；Full400完成后的CP4汇报标记仍须届时实际核查，不由本次预检提前认可。

15:00的[CPU跟进采样](../diagnostics/resource-context/20261111T070002281685Z-cpu-pressure.json)仍记录到load1为1986.5、CPU some PSI avg10为77.44%；最大外部用户在1.337秒采样中占用约373.7个逻辑核。CPU最高的15个进程中有13个来自同一父进程，这13个进程各有194–257线程。该采样与全用户统计的范围不同，不能把后者全部归入这个父进程。同一时点的[GPU队列核查](../diagnostics/resource-context/20261111T070001699814Z-shared-queue.json)显示GPU1、2、3各有两个本实验计算进程和两个外部计算进程，外部占用各4016 MiB；三个本实验场景均在运行，没有空闲卡等待领取。两次查询均未发送信号或修改原生任务，GPU0继续保留给用户。

15:58的[CPU采样](../diagnostics/resource-context/20261111T075846904195Z-cpu-pressure.json)中，load1为1516.85，CPU some PSI avg10为67.45%，最大外部用户仍占用约371.7个逻辑核。最高15个CPU进程中有9个同属一个父进程，各有194–257线程；仍分别报告这组采样与全用户统计。同期[GPU队列核查](../diagnostics/resource-context/20261111T075846042206Z-shared-queue.json)确认三个本实验场景均在运行。相比此前采样，负载和等待压力有所下降，但资源拥堵仍未解除；不据此推断用户已完成线程协调或恢复原定完成时间。

16:57的[CPU采样](../diagnostics/resource-context/20261111T085727532421Z-cpu-pressure.json)记录到load1为936.21、CPU some PSI avg10为52.98%，最大外部用户采样占用约340.5个逻辑核。[同期队列快照](../diagnostics/resource-context/20261111T085726772376Z-shared-queue.json)中，GPU3已领取p53SfW6mjZe，空闲41058 MiB、已有外部compute占用4016 MiB，满足原有38 GiB空闲和不超过8 GiB已有compute的门槛；当时NVML尚未记录到新场景的计算进程。

cvZr5TUy5C5和mL8ThkuaVTM均完成各自20条固定回合后正常退出，控制器分别于16:57:21和17:03:46捕获Linux wait status=0，再回收停止的CPU父进程704760、703335。[CV核查](../diagnostics/gpu-resource-sharing/20261111T052457548416Z/native-completion-audits/20261111T090843197451Z-cvZr5TUy5C5.json)和[ML核查](../diagnostics/gpu-resource-sharing/20261111T052457548416Z/native-completion-audits/20261111T091146708094Z-mL8ThkuaVTM.json)复核原生completed状态、20条身份、三目标计数、退出记录与父进程回收记录，并绑定当前共享调度源码和冻结参数。两次只读核查均未发送信号或修改原生状态。原/proc记录已随父进程回收，本次使用控制器此前捕获的记录，独立OS重读标记均为false。回合结束和进程退出码0不表示所有导航目标成功；失败阶段和预算耗尽结果照常保留。

GPU3随后运行p53SfW6mjZe，GPU1运行q3zU7Yy5E5s。17:08的[启动审计](../diagnostics/full/parallel-scene-launch-audits/20261111T090843336198Z.json)核对15个已启动场景的300条身份、类别顺序、source ID、seed=1和运行源码；[环境索引](../diagnostics/environment-index/20261111T090843204115Z.json)核对20条冒烟和15个Full场景的双角色记录。同期直接查询已见两个新场景0号回合分别有507和314个实际导航动作。p53SfW6mjZe的[首条完整回合](../diagnostics/completed-prefix/full/p53SfW6mjZe/0.json)随后以787步结束，成功向量为1/1/1，三个阶段的路径、身份及冻结参数核查通过。其余5个场景仍按原顺序排队。

17:15的[CPU采样](../diagnostics/resource-context/20261111T091548330767Z-cpu-pressure.json)中，load1降至341.5，最大外部用户采样占用约97.4个逻辑核，CPU some PSI avg10仍为83.68%。load和等待压力反映不同方面，不能仅据load下降判断拥堵解除。后续比较保留本段完整结果、规划时延及资源阶段记录，完成时间继续依据实际吞吐更新。

18:34的[CPU采样](../diagnostics/resource-context/20261111T103446803764Z-cpu-pressure.json)记录到load1为542.95、CPU some PSI avg10为57.03%，最大外部用户在该次采样中占用约175.3个逻辑核。这些是主机及全用户层面的离散统计，不用于判断某一任务组的线程限制是否已生效；完整原生结果与规划时延继续保留。

18:54:43，mv2HUxq3B53完成固定20条回合，当前控制器通过其持有的subprocess.Popen.poll取得退出码0。18:57的[队列快照](../diagnostics/resource-context/20261111T105734168023Z-shared-queue.json)中，GPU2已经没有本实验计算进程，但外部compute占用8280 MiB、空闲36354 MiB，未满足既定启动门槛，下一场景等待；GPU1、3继续运行。

19:11的[队列复查](../diagnostics/resource-context/20261111T111139683407Z-shared-queue.json)确认GPU2已领取qyAac8rV8Zk，三张授权卡均有两个本实验计算进程，没有外部compute进程或空闲槽位等待。启动门槛、原生源码、冻结参数和任务身份均未改变。[启动审计](../diagnostics/full/parallel-scene-launch-audits/20261111T111149097034Z.json)已覆盖16个场景的320条固定身份；[环境索引](../diagnostics/environment-index/20261111T111158217842Z.json)核对20条冒烟和16个Full场景的双角色记录，剩余4个场景仍待启动。

19:12的[CPU采样](../diagnostics/resource-context/20261111T111208325531Z-cpu-pressure.json)中，load1为10.73，CPU some PSI avg10、avg60均为0，最大外部用户采样占用约0.027个逻辑核。此前协调的进程组在本次只读采样中已不存在，主机等待压力明显缓解；采样没有核实具体的线程限制操作或任务结束原因。本次资源恢复与用户协调回复分别记录，不据此作因果归因。所有原生回合和规划时延仍完整保留，本实验未向外部任务发送信号，GPU0继续保留给用户。

20:18:58，p53SfW6mjZe完成固定20条回合，当前控制器持有的subprocess.Popen.poll记录退出码0；20:31的只读复查核对原生completed状态及20条身份。GPU3随后运行svBbv1Pavdk，20:32的[启动审计](../diagnostics/full/parallel-scene-launch-audits/20261111T123203482175Z.json)已覆盖17个场景、340条固定身份，目标顺序、source ID、seed=1与运行源码一致。[环境索引](../diagnostics/environment-index/20261111T123211438645Z.json)核对20条冒烟及17个Full场景的双角色记录，剩余3个场景尚未启动。

20:32的[队列核查](../diagnostics/resource-context/20261111T123222827956Z-shared-queue.json)确认GPU1、2、3各有两个本实验计算进程，外部compute均为0，没有空闲槽位等待资源。三卡继续运行原生任务，原有启动显存门槛、冻结参数、源码和GPU授权范围保持不变。

新场景svBbv1Pavdk的[首条完整回合](../diagnostics/completed-prefix/full/svBbv1Pavdk/0.json)以1500步结束，三阶段成功向量为0/0/0。身份、冻结参数、各阶段终止STOP和实际路径归属已核对，三个失败阶段均原样保留。

20:48:10，q3zU7Yy5E5s的固定20条回合完成，当前控制器的subprocess.Popen.poll记录退出码0，随后只读复核原生completed状态与20条身份。GPU1接续wcojb4TFT35；20:52的[启动审计](../diagnostics/full/parallel-scene-launch-audits/20261111T125244146244Z.json)核对18个已启动场景、360条固定身份，[环境索引](../diagnostics/environment-index/20261111T125253167358Z.json)核对20条冒烟和18个Full场景的双角色记录。剩余ziup5kvtCCR、zt1RVoi7PcG两个场景尚未启动，后续仍按原顺序领取。

20:53的[队列快照](../diagnostics/resource-context/20261111T125313584802Z-shared-queue.json)记录到新的共享占用：GPU1、2、3各有4个外部compute进程、共8280 MiB，同时各保留两个本实验计算进程，没有空闲槽位等待资源。20:54的[CPU采样](../diagnostics/resource-context/20261111T125432058649Z-cpu-pressure.json)中，384个逻辑核上的load1为196.4，CPU some/full PSI的avg10、avg60、avg300均为0；最大外部用户采样占用约180个逻辑核。最高15个CPU进程同属一个父进程，每个11线程；该组采样与全用户统计分开报告。本时点未观察到CPU等待压力。

现有场景继续运行，38 GiB空闲及不超过8 GiB已有compute占用的领取门槛保持原值。8280 MiB已超过后一门槛；后续如仍保持这项占用，空出的授权卡将等待资源。只读范围检查发现该外部进程组同时占用0至7号GPU，因而没有向该组或其worker发送停止信号。GPU0继续保留给用户，所有已完成回合和时延记录照常保留。

wcojb4TFT35的[首条完整回合](../diagnostics/completed-prefix/full/wcojb4TFT35/0.json)随后以1500步结束，成功向量为0/0/0；三阶段路径、原生STOP、身份及冻结参数均通过记录核查。所有阶段保留，未以这条结果调整参数或重跑失败回合。

21:22的[资源复查](../diagnostics/resource-context/20261111T132224311264Z-shared-queue.json)仍记录到三张授权卡各4个外部compute进程、8280 MiB占用，三个本实验场景都在运行，无空闲槽位等待。再次只读核查确认该外部组覆盖0至7号GPU，没有发送信号。已向用户提出协调减少GPU1、2、3共享占用的请求，现有实验继续；请求不视为已经释放显存，后续按实测资源决定领取。

22:06的[资源复查](../diagnostics/resource-context/20261111T140602229390Z-shared-queue.json)确认GPU1、2、3的外部compute进程数及显存占用均为0，各卡仍有两个本实验计算进程，三个原生场景继续运行，没有空闲槽位等待资源。此前提出的共享显存协调请求尚未收到用户答复；本次仅确认请求后观察到资源释放，未核实具体协调操作或释放原因。原启动门槛、冻结参数和GPU授权范围保持不变，本实验未向外部进程发送信号。所有回合和规划时延继续保留。

22:35:10，svBbv1Pavdk完成固定20条回合，当前控制器通过其持有的subprocess.Popen.poll记录退出码0；22:36的只读复查核对completed状态、20条身份覆盖及原生状态文件SHA。GPU3随后接续ziup5kvtCCR。22:37的[启动审计](../diagnostics/full/parallel-scene-launch-audits/20261111T143717155226Z.json)已核对19个已启动场景的380条固定身份、类别顺序、source ID、seed=1及源码，[环境索引](../diagnostics/environment-index/20261111T143717596752Z.json)覆盖20条冒烟和19个Full场景的双角色环境。新场景已开始产生实际导航动作，仅zt1RVoi7PcG尚未启动；新场景的首条完整回合GT审计仍待其原生指标导出后进行。原启动门槛、GPU1/2/3分配和冻结参数保持不变，未停止或重启其他原生场景。

22:55的增量GT审计已覆盖新场景ziup5kvtCCR的[首条完整回合](../diagnostics/completed-prefix/full/ziup5kvtCCR/0.json)：830步结束，成功向量1/1/0；[第二条回合](../diagnostics/completed-prefix/full/ziup5kvtCCR/1.json)为662步、1/1/1。两条的三阶段原生STOP、实际路径、候选路径归属、身份和冻结参数均通过核查，首条的失败阶段照常保留。本次Full过程审计已达352/400条、1056个原生阶段，未修改运行源码、参数或重跑失败回合。

23:04:02，qyAac8rV8Zk完成固定20条回合，当前控制器通过其持有的subprocess.Popen.poll记录退出码0；23:05只读复查核对completed状态、20条覆盖和原生status文件SHA。GPU2随后接续最后一个待启动场景zt1RVoi7PcG。23:07的[启动审计](../diagnostics/full/parallel-scene-launch-audits/20261111T150733113327Z.json)覆盖全部20个场景的400条固定身份、顺序类别、source ID、seed=1和运行源码，[环境索引](../diagnostics/environment-index/20261111T150733690782Z.json)覆盖20条冒烟及20个Full场景的双角色环境。最后一个场景已产生实际导航动作，首条完整回合GT仍待原生指标导出后核查。全部场景已启动不表示全部400条已完成；仍继续运行每条固定回合，全部失败结果与规划时延保留。仅使用GPU1、2、3，没有修改源码、冻结参数或停止其他场景。

23:12的增量GT审计已覆盖最后一个场景zt1RVoi7PcG的[首条完整回合](../diagnostics/completed-prefix/full/zt1RVoi7PcG/0.json)：302步结束，三阶段成功向量1/1/1，原生STOP、实际路径及候选路径归属、身份和冻结参数核查均通过。Full过程审计至此覆盖360/400条、1080个原生阶段；其余回合继续在线运行。全程保留失败和预算耗尽结果，没有根据首条结果修改算法或参数。

23:14的[匿名资源快照](../diagnostics/resource-context/20261111T151422002111Z-shared-queue.json)记录到GPU1有2008 MiB外部compute占用，GPU2、3为0；三个原生场景仍在运行，已无未启动场景或空槽等待。该离散记录未确定外部任务身份，也不用于推断其对速度或成功率的影响。原启动门槛、GPU1/2/3分配与冻结参数保持不变，未向外部任务发送信号，全部回合和时延继续保留。

11月12日00:19:35，ziup5kvtCCR完成固定20条回合，当前控制器通过其持有的subprocess.Popen.poll记录退出码0；00:21的只读复查核对原生completed状态、20条身份覆盖及status文件SHA。Full已无未启动场景，因此GPU3的本轮场景结束后不再领取新Full任务，GPU1、2继续各自的原生场景。后续变体仍按Full完整400条、实际CP4汇报后依次启动，不跨变体并发。本次回合和进程完成均为原生正常结束，没有停止或重跑工作进程，全部失败与时延记录继续保留。

11月12日00:32:11，wcojb4TFT35完成固定20条回合，当前控制器通过其持有的subprocess.Popen.poll记录退出码0；00:34只读复查核对20条原生身份及status文件SHA。Full至此已有19个完整场景，最后的zt1RVoi7PcG继续在GPU2运行；GPU1、3均因各自场景正常结束而空出。Full没有未启动场景，后续变体仍在Full400及实际CP4汇报后依次启动。没有停止或重跑原生工作进程，全部失败回合和资源时延记录保留。

11月12日01:19:51，最后一个Full场景zt1RVoi7PcG完成20条回合，原控制器持有的subprocess.Popen.poll记录退出码0；01:20只读复查核对20条原生身份和status文件SHA。至此20个场景均正常退出，固定400条原生结果全部导出，随后由原控制器执行全量GT与指标汇总。GPU1、2、3的Full任务均自然结束，没有通过停止、剔除或重跑失败回合改变结果。后续变体仍须在完整CP4核查、发布并实际汇报后启动，GPU0继续留给用户。

Nearest首次尝试因物理GPU3的mapper CUDA OOM中断。此次57条整体归档，随后完整400条重新运行；具体记录和原始SHA见[恢复记录](Nearest_infrastructure_restart.md)。用户另行授权GPU3的其他用户进程清理与20秒周期监督，GPU0及4至7不作为信号目标。

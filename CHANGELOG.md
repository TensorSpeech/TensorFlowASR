# CHANGELOG

<!-- version list -->

## v3.0.0 (2025-05-25)

### Bug Fixes

- Add ctc tpu impl
  ([`82d91c8`](https://github.com/TensorSpeech/TensorFlowASR/commit/82d91c844ce418c935fa82b02ac93a7fe690b097))

- Add custom batch norm to avoid tf.cond
  ([`b5bfe92`](https://github.com/TensorSpeech/TensorFlowASR/commit/b5bfe920b53116bae4606a32b9ad7ce662982a07))

- Add setup.sh script, update make function
  ([`64aeb2f`](https://github.com/TensorSpeech/TensorFlowASR/commit/64aeb2f11b6b8f9219a3e09800bdedd8cb50e97d))

- Add sync batch norm, remove wrong bn in ds2
  ([`7c21c1d`](https://github.com/TensorSpeech/TensorFlowASR/commit/7c21c1dac46d29b44ecd5ff596ec1cc89a6e2d98))

- Allow ctc to force use native tf impl
  ([`a5d1e84`](https://github.com/TensorSpeech/TensorFlowASR/commit/a5d1e84f38a9cd7de27560052b162e940820362d))

- Apply ga loss division before loss scaling
  ([`dde7760`](https://github.com/TensorSpeech/TensorFlowASR/commit/dde77604567815cbe3587c2412a4d509808f0731))

- Attention mask
  ([`3dbab33`](https://github.com/TensorSpeech/TensorFlowASR/commit/3dbab3352b3b7e23ad6f64fadf7b984991a2d1c8))

- Backup and restore callback
  ([`38f641c`](https://github.com/TensorSpeech/TensorFlowASR/commit/38f641cd1d733da9b86b2317e591e41b08e7d93f))

- Callback
  ([`7d3c954`](https://github.com/TensorSpeech/TensorFlowASR/commit/7d3c9544fa36af0c98f544cdff034ba13dd162fa))

- Config
  ([`6cba25e`](https://github.com/TensorSpeech/TensorFlowASR/commit/6cba25ee5a7c8050de88d910550cb5f8d3e64d24))

- Config
  ([`35feaaf`](https://github.com/TensorSpeech/TensorFlowASR/commit/35feaaf23966a8731e1ffa08df2624749e20fc0e))

- Config
  ([`c15e841`](https://github.com/TensorSpeech/TensorFlowASR/commit/c15e8413df73e545d0c072af945163c5e4dea803))

- Config
  ([`98390fc`](https://github.com/TensorSpeech/TensorFlowASR/commit/98390fc8898d78da423b64d8743b43f400d480bb))

- Config
  ([`e88e99a`](https://github.com/TensorSpeech/TensorFlowASR/commit/e88e99a9440e86beb4731f12e47e3ebc4b4eaca9))

- Config streaming
  ([`70ac41e`](https://github.com/TensorSpeech/TensorFlowASR/commit/70ac41e5d46659ced697f4b02fe9b99adddc0dae))

- Configs
  ([`aa15483`](https://github.com/TensorSpeech/TensorFlowASR/commit/aa1548381c8ff31e98335bc114ab300e540588e1))

- Configs
  ([`57aef49`](https://github.com/TensorSpeech/TensorFlowASR/commit/57aef49ac329c7568511bd4db163d8bf5a790236))

- Configs
  ([`6b0bec4`](https://github.com/TensorSpeech/TensorFlowASR/commit/6b0bec4e608d719c5b8e1f83e79b50bff09db215))

- Configs
  ([`eebc361`](https://github.com/TensorSpeech/TensorFlowASR/commit/eebc36165ffe5f39c4b7bea6b267c7dee5ea6a17))

- Configs, add gradn step
  ([`2e2d6e4`](https://github.com/TensorSpeech/TensorFlowASR/commit/2e2d6e41f33913cecc9dfc91032b885c1dd37118))

- Conformer ctc
  ([`4e75c0f`](https://github.com/TensorSpeech/TensorFlowASR/commit/4e75c0f2847855272a85dada604752e03ea4d1e3))

- Conformer ctc configs
  ([`a302962`](https://github.com/TensorSpeech/TensorFlowASR/commit/a302962d80ec81d29237918e15028fff09bbd8bf))

- Conformer ctc decoder
  ([`422d4bb`](https://github.com/TensorSpeech/TensorFlowASR/commit/422d4bb8b397be24300028e813d1514b2152a448))

- Contextnet
  ([`8b0ed02`](https://github.com/TensorSpeech/TensorFlowASR/commit/8b0ed028c6fac628fcaac952d4b292a8b1adf32d))

- Ctc
  ([`9b46cbe`](https://github.com/TensorSpeech/TensorFlowASR/commit/9b46cbe797658e68df981bbeb46f19131fa9fa18))

- Ctc loss
  ([`ebb6930`](https://github.com/TensorSpeech/TensorFlowASR/commit/ebb69302cfe858e4b083e21294875c9cd9404eba))

- Ctc loss tpu
  ([`f1a0ed6`](https://github.com/TensorSpeech/TensorFlowASR/commit/f1a0ed6df68ff4d8bbb1bb30368b67535b2a77f6))

- Ctc loss tpu - case logits to float32
  ([`778c1a2`](https://github.com/TensorSpeech/TensorFlowASR/commit/778c1a294602502e33b12ed1acf37ed6887c9859))

- Ctc tpu impl
  ([`c915be3`](https://github.com/TensorSpeech/TensorFlowASR/commit/c915be3f3a49ca1007d883f2cabefeda01384e5d))

- Ctc-tpu clean label
  ([`c35af45`](https://github.com/TensorSpeech/TensorFlowASR/commit/c35af45504ce9376da29f717b48f358e7618fb62))

- Deps
  ([`1dc854e`](https://github.com/TensorSpeech/TensorFlowASR/commit/1dc854e2e6459a88e32d90e4c8cf3ff6c3b81314))

- Deps
  ([`1ae5a66`](https://github.com/TensorSpeech/TensorFlowASR/commit/1ae5a6682058c549a8b39c6ec4542dc6e7fa663b))

- Deps
  ([`6ca9c61`](https://github.com/TensorSpeech/TensorFlowASR/commit/6ca9c618d280950a088f648181183b33f8940772))

- Deps
  ([`5e31671`](https://github.com/TensorSpeech/TensorFlowASR/commit/5e316710ce89b349866a634671f28b7b15d8f37e))

- Deps
  ([`cea79ef`](https://github.com/TensorSpeech/TensorFlowASR/commit/cea79ef39ffb99f5530c321b7d739432e3e740cf))

- Deps
  ([`36f3cc0`](https://github.com/TensorSpeech/TensorFlowASR/commit/36f3cc0db5d2a21c0ab84607b224302e493cfa27))

- Deps
  ([`6dc3410`](https://github.com/TensorSpeech/TensorFlowASR/commit/6dc3410c1aabc69259735fe54ac3d41ca1eb51e9))

- Deps
  ([`c532d5c`](https://github.com/TensorSpeech/TensorFlowASR/commit/c532d5c6c8219ad65c8d9f9d43fe2e2d964763d8))

- Disable bias/activity regularizer as not needed
  ([`d538e69`](https://github.com/TensorSpeech/TensorFlowASR/commit/d538e69f744bff1b47fb2571ae015a59938d8c63))

- Disable tqdm, logging for kagglehub
  ([`23dd668`](https://github.com/TensorSpeech/TensorFlowASR/commit/23dd66884fa5ea1c60427f7f7a3a7c16178fe68f))

- Ds2
  ([`77baaa5`](https://github.com/TensorSpeech/TensorFlowASR/commit/77baaa526f6564ff6bf246210334fe3f22b2d5d2))

- Env util
  ([`4390d1b`](https://github.com/TensorSpeech/TensorFlowASR/commit/4390d1b2ac571df988bea5a97ecd9f120266c0f9))

- Env utils
  ([`dc08e6a`](https://github.com/TensorSpeech/TensorFlowASR/commit/dc08e6ab16c60fd4e7b654bca876688b9c9dc012))

- Expose relmha_causal, flash attention
  ([`2100d75`](https://github.com/TensorSpeech/TensorFlowASR/commit/2100d752ca43cc2d60e77a25242421bd69d27181))

- Feature extraction layer dtype
  ([`a2eaf15`](https://github.com/TensorSpeech/TensorFlowASR/commit/a2eaf15a725ca65e2b5b4e3e8eb75135c35d5bf5))

- Feature extraction layer dtype tf.float32 to ensure loss convergence
  ([`c3ab865`](https://github.com/TensorSpeech/TensorFlowASR/commit/c3ab865d89ca3ff43a918b3b040bb8ff69c66459))

- Feature extraction mixed precision, configs
  ([`12fbb85`](https://github.com/TensorSpeech/TensorFlowASR/commit/12fbb8548ddd4096fa0e375198294f22b00be695))

- File util
  ([`63f6280`](https://github.com/TensorSpeech/TensorFlowASR/commit/63f628070dbb4249d1ca59ff5e93971b1247012d))

- Ga
  ([`522f080`](https://github.com/TensorSpeech/TensorFlowASR/commit/522f0802a8ea7a73dfcc2bfc0b22c89ef6c44967))

- Ga
  ([`8285f6d`](https://github.com/TensorSpeech/TensorFlowASR/commit/8285f6dc9ccc1112e3a1d05ab7eeaca7be097e30))

- General layers to show outputshape, invalid loss show outputs
  ([`1545543`](https://github.com/TensorSpeech/TensorFlowASR/commit/154554308dcb73ffc80c8391c93146a888d1e92d))

- Gradient accumulation
  ([`1eb2d15`](https://github.com/TensorSpeech/TensorFlowASR/commit/1eb2d15220f4630730123cd0fc967883e7518a61))

- Gradn
  ([`d36085e`](https://github.com/TensorSpeech/TensorFlowASR/commit/d36085e284f316b70f56cd38c3e2f6d44ce380d6))

- Handle unknown dataset size with no metadata provided
  ([`ba9d6b2`](https://github.com/TensorSpeech/TensorFlowASR/commit/ba9d6b2d604f812e3359937948d0184dfe3ff54c))

- Ignore backup kaggle when nan loss occurs
  ([`f62fdf9`](https://github.com/TensorSpeech/TensorFlowASR/commit/f62fdf923475cc5563b66b6875fe706672a94984))

- Jasper
  ([`3768878`](https://github.com/TensorSpeech/TensorFlowASR/commit/37688784931d0e9de79c746cf82531867e50d1f4))

- Kaggle backup & restore callback
  ([`2d557cc`](https://github.com/TensorSpeech/TensorFlowASR/commit/2d557cc1f9983051f82cd902818c8f0977400784))

- Log batch that cause invalid loss
  ([`cf37206`](https://github.com/TensorSpeech/TensorFlowASR/commit/cf37206a4cee95ad791e40f3dad679dc4f90923b))

- Loss compute using add_loss, loss tracking
  ([`f8a7b91`](https://github.com/TensorSpeech/TensorFlowASR/commit/f8a7b9104e2274f572a18e0e0c779fa1297acc4d))

- Make function
  ([`a9218d9`](https://github.com/TensorSpeech/TensorFlowASR/commit/a9218d920407e63d987a96e77b4727d423512f77))

- Mha streaming mask
  ([`eb55a4b`](https://github.com/TensorSpeech/TensorFlowASR/commit/eb55a4baed729db8e10e16d905f0d2c29e837d26))

- Models configs
  ([`5f784b7`](https://github.com/TensorSpeech/TensorFlowASR/commit/5f784b78e79d2c5c4a2c68b65b8cdeb3470a5632))

- Nan to num
  ([`f2241cf`](https://github.com/TensorSpeech/TensorFlowASR/commit/f2241cf0334957bcdd4cc3546dee0e474ee97647))

- Nan to num
  ([`2f502bb`](https://github.com/TensorSpeech/TensorFlowASR/commit/2f502bb1f9d6957a832879a59827a4af8eb512c3))

- Numeric stability with dtype compatible
  ([`5a1f054`](https://github.com/TensorSpeech/TensorFlowASR/commit/5a1f05489b209ecd4c3cec2cba9175e888b2df63))

- Only use tqdm when needed
  ([`2947160`](https://github.com/TensorSpeech/TensorFlowASR/commit/2947160782ec682d2c8d0e555ca5d0b38b3f2a5b))

- Only wrap tf.function in jit compile
  ([`4e0e8f5`](https://github.com/TensorSpeech/TensorFlowASR/commit/4e0e8f586363423b363fece886f0797b72ad0aff))

- Option TF_CUDNN
  ([`f4e459d`](https://github.com/TensorSpeech/TensorFlowASR/commit/f4e459ded8e8d3e16a61955b061e1a72f6e20c03))

- Output shapes of models to log to summary
  ([`03f0d60`](https://github.com/TensorSpeech/TensorFlowASR/commit/03f0d6042470dde4093b7fd497cbfbe5e1331d64))

- Pad logits length to label length
  ([`f1e2a88`](https://github.com/TensorSpeech/TensorFlowASR/commit/f1e2a881d79d3596908b142ec28911a8217b1e9f))

- Prediction funcs, initial states
  ([`79eb87d`](https://github.com/TensorSpeech/TensorFlowASR/commit/79eb87d01160e0c4e5a35c33af64ef32f55b1f0e))

- Print shapes
  ([`689b366`](https://github.com/TensorSpeech/TensorFlowASR/commit/689b366917bf904df081ff23833a66f5e2e058f4))

- Remove ds2 dropout on conv module
  ([`9ea9b87`](https://github.com/TensorSpeech/TensorFlowASR/commit/9ea9b8740f289580361f05f278b7db4b70c69981))

- Requirements
  ([`20d30c6`](https://github.com/TensorSpeech/TensorFlowASR/commit/20d30c62be5dd19eea12f1f3c6f28c8277ec06a0))

- Requirements
  ([`e524af6`](https://github.com/TensorSpeech/TensorFlowASR/commit/e524af6f6f5ed028d1e9e2a26e314f87d1528caf))

- Requirements
  ([`ace3887`](https://github.com/TensorSpeech/TensorFlowASR/commit/ace3887fb3c7ec65f2f59cb76ee81c3e38843fcf))

- Requirements
  ([`9b03b31`](https://github.com/TensorSpeech/TensorFlowASR/commit/9b03b313b293eca9a772730e1dbfe21f250f439b))

- Requirements with docker
  ([`7f145f7`](https://github.com/TensorSpeech/TensorFlowASR/commit/7f145f7a666f4c7206624a5311edd7419dd288e4))

- Restore from kaggle model
  ([`6e5c3b9`](https://github.com/TensorSpeech/TensorFlowASR/commit/6e5c3b9ceede88315b8e9b030c2937e639c9161b))

- Restore from kaggle model
  ([`2fd4f2b`](https://github.com/TensorSpeech/TensorFlowASR/commit/2fd4f2bb0506d8d0ce82648dd67fa8a7d5b6f974))

- Rnn kwargs
  ([`de58fed`](https://github.com/TensorSpeech/TensorFlowASR/commit/de58fedf94a5b3ae6e7d2988eaae074841c9ff07))

- Rnnt
  ([`f3cb239`](https://github.com/TensorSpeech/TensorFlowASR/commit/f3cb239af8f85fb90483ba2c54f025794bd6469e))

- Save weights, tpu connect
  ([`c667984`](https://github.com/TensorSpeech/TensorFlowASR/commit/c667984028e24d6eda9f88dc5b2820d521feba96))

- Save weights, tpu connect
  ([`dda33b7`](https://github.com/TensorSpeech/TensorFlowASR/commit/dda33b7d91c8e37e633ded94bde19cc55039c0dc))

- Scripts
  ([`bb3b848`](https://github.com/TensorSpeech/TensorFlowASR/commit/bb3b848d06734da8799b0af735dcc0073e27cbe1))

- Setup strategy with set visible devices
  ([`0e13f76`](https://github.com/TensorSpeech/TensorFlowASR/commit/0e13f76132d2bd09cd18e26a763df11ebb000f89))

- Small kaggle
  ([`454163c`](https://github.com/TensorSpeech/TensorFlowASR/commit/454163c3c9ccea0c81358f1b1b73383326106358))

- Soft device placement
  ([`90e6695`](https://github.com/TensorSpeech/TensorFlowASR/commit/90e6695dc84f44820ce60972ad9503bb25e7865c))

- Softmax numberic overflow with mask
  ([`edf5eb2`](https://github.com/TensorSpeech/TensorFlowASR/commit/edf5eb298c5a8402cc6e14c8b0a859d68e8bf742))

- Strategy scope
  ([`9179425`](https://github.com/TensorSpeech/TensorFlowASR/commit/91794250d02ba195082942f0d06c8f14461bdff9))

- Streaming masking mha
  ([`e209040`](https://github.com/TensorSpeech/TensorFlowASR/commit/e20904061c49df14a99915b779913be677e71369))

- Streaming masking mha
  ([`7dcd145`](https://github.com/TensorSpeech/TensorFlowASR/commit/7dcd1457c26dcc2a7e2fab548a9be34ac09bb308))

- Streaming masking mha
  ([`eca664c`](https://github.com/TensorSpeech/TensorFlowASR/commit/eca664c7c2052132bcd324b1928246e829a43b67))

- Super init
  ([`401180b`](https://github.com/TensorSpeech/TensorFlowASR/commit/401180b2eedbbb9ed5db1a455d2210ba652578a9))

- Support flash attention, update deps
  ([`52de4c0`](https://github.com/TensorSpeech/TensorFlowASR/commit/52de4c0d65a6d680093785d3afa485224d83d654))

- Support log debug
  ([`7aed458`](https://github.com/TensorSpeech/TensorFlowASR/commit/7aed4580b422f2696ef485fc38409239650ae4de))

- Support static shape
  ([`0e5e826`](https://github.com/TensorSpeech/TensorFlowASR/commit/0e5e826b43ae9664540f61f550c088d55cfd1a72))

- Tflite, initial states, results
  ([`db66008`](https://github.com/TensorSpeech/TensorFlowASR/commit/db66008b3d2bbc4245aaf4d4618df64ccba11bfd))

- Train script
  ([`6a90a51`](https://github.com/TensorSpeech/TensorFlowASR/commit/6a90a518a4090d8b23ecbf2e4e7c87673b6f2210))

- Train step
  ([`8ee9813`](https://github.com/TensorSpeech/TensorFlowASR/commit/8ee9813256abadb518edc71a909d3f58d866dfc7))

- Transformer
  ([`1d7e3a6`](https://github.com/TensorSpeech/TensorFlowASR/commit/1d7e3a67255e246b97e35dca9ee7624de7698e35))

- Update compute mask ds2
  ([`7441c95`](https://github.com/TensorSpeech/TensorFlowASR/commit/7441c95f9c0eaba724c2823c5f54b28d2d481725))

- Update config
  ([`66b394e`](https://github.com/TensorSpeech/TensorFlowASR/commit/66b394ec8fe23104ec19889ce7150c3291671c33))

- Update deps
  ([`7be3eda`](https://github.com/TensorSpeech/TensorFlowASR/commit/7be3eda46b46c66d7fc74111c0b840df5f91260c))

- Update docs and results
  ([`310fde4`](https://github.com/TensorSpeech/TensorFlowASR/commit/310fde4732a85b3d298a700c29fab45e9669e601))

- Update gradient accumulation
  ([`c8d3614`](https://github.com/TensorSpeech/TensorFlowASR/commit/c8d361418c3519630c3934b538dc2a609d7ad148))

- Update gradient accumulation
  ([`4330dec`](https://github.com/TensorSpeech/TensorFlowASR/commit/4330dec4dee100efac99ee9e7c7ad938cb5b30ce))

- Update kaggel callback
  ([`65fdf46`](https://github.com/TensorSpeech/TensorFlowASR/commit/65fdf46af722b91e79f4a1d84611f53c9f266ee6))

- Update masking and layer
  ([`51a5258`](https://github.com/TensorSpeech/TensorFlowASR/commit/51a5258c2330cd76bd28d177dc38939bbbd3cfd2))

- Update masking and layer
  ([`076a6f9`](https://github.com/TensorSpeech/TensorFlowASR/commit/076a6f994557bcb6beadb8a19dd8895b754b150c))

- Update mha attention mask
  ([`5649fdd`](https://github.com/TensorSpeech/TensorFlowASR/commit/5649fdd9b2f04e36dc8d1412137042cc0ae5c4c5))

- Update regularizers
  ([`a9d1733`](https://github.com/TensorSpeech/TensorFlowASR/commit/a9d17330c4308a1a3b04d73b2f182270b8127cfd))

- Update regularizers
  ([`786f5d4`](https://github.com/TensorSpeech/TensorFlowASR/commit/786f5d43836f03e651356f010e0f19608cacd718))

- Update req
  ([`bb732a7`](https://github.com/TensorSpeech/TensorFlowASR/commit/bb732a7d8af1557d59c9912cfa92e394b5ad4a79))

- Update req
  ([`6ffb3b8`](https://github.com/TensorSpeech/TensorFlowASR/commit/6ffb3b8ba7a89639e0524b7e09816959fb692bde))

- Update req
  ([`35160ce`](https://github.com/TensorSpeech/TensorFlowASR/commit/35160ce9b2dbd67f8a21a43fe1c2368cae77c313))

- Update req
  ([`33394a2`](https://github.com/TensorSpeech/TensorFlowASR/commit/33394a2992e8b3126af98bac64ccb6822802e5ea))

- Update req
  ([`d455ae1`](https://github.com/TensorSpeech/TensorFlowASR/commit/d455ae1cf273dac399f8ad1501bfa715ef2a6cd6))

- Update req
  ([`dc77b84`](https://github.com/TensorSpeech/TensorFlowASR/commit/dc77b847c14e656b6a3758f6e132b12df7fb6740))

- Update savings
  ([`67a8470`](https://github.com/TensorSpeech/TensorFlowASR/commit/67a847069692ef26433074b3b5b04bf3177e19f8))

- Update train function with ga steps
  ([`aade071`](https://github.com/TensorSpeech/TensorFlowASR/commit/aade07187e744b221c5d309d341f7c126bf9ea44))

- Update train step
  ([`305ddab`](https://github.com/TensorSpeech/TensorFlowASR/commit/305ddabd4ccd55bff53f384c2c3c13c83b539a33))

- Update train step ga
  ([`dc0c304`](https://github.com/TensorSpeech/TensorFlowASR/commit/dc0c304d45cc070a9c9d38872ba5fbf1f1d08722))

- Update train/test step
  ([`d55fd40`](https://github.com/TensorSpeech/TensorFlowASR/commit/d55fd400182ef5ddf5fc9718dc132f617ecd6592))

- Update vocab generator
  ([`7d2f029`](https://github.com/TensorSpeech/TensorFlowASR/commit/7d2f029622f27ea73acb6f8d6710e91e26610c90))

- Update vocab generator
  ([`b325143`](https://github.com/TensorSpeech/TensorFlowASR/commit/b32514346ac20d4f4614fe1a596842c06e47683b))

- Update vocab generator
  ([`c5024ac`](https://github.com/TensorSpeech/TensorFlowASR/commit/c5024ac8989cd8e1c63df81da80479ac5244b268))

- Use auto mask
  ([`e68ceee`](https://github.com/TensorSpeech/TensorFlowASR/commit/e68ceee25b467941450e84174286397afce70aa5))

- Use autograph do_not_convert for batchnorm sync to work
  ([`de38407`](https://github.com/TensorSpeech/TensorFlowASR/commit/de3840736931fe0c849bea7459eaa989f647d678))

- Use default make function
  ([`ce6752b`](https://github.com/TensorSpeech/TensorFlowASR/commit/ce6752b46b46d0123589f7c7fb00ddb284738a83))

- Use history size instead of memory length
  ([`a2e2022`](https://github.com/TensorSpeech/TensorFlowASR/commit/a2e20229ab7f99793750541adbced410e7bf7b89))

- Use keras-nightly
  ([`f5886a5`](https://github.com/TensorSpeech/TensorFlowASR/commit/f5886a5fb08bef026ad45c91c7f56c5574f75dc1))

### Chores

- Add conformer small streaming
  ([`0543c31`](https://github.com/TensorSpeech/TensorFlowASR/commit/0543c31fef0f393ba956b4819f78b0f8d91ecd61))

- Add conformer small streaming
  ([`e844f77`](https://github.com/TensorSpeech/TensorFlowASR/commit/e844f773871c883cb7a42c002197b5d994fd3774))

- Add conformer-ctc-small-streaming-kaggle
  ([`aaa06a5`](https://github.com/TensorSpeech/TensorFlowASR/commit/aaa06a5c47d66f5932d580cb4e42ecf4bc841ac9))

- Add option use loss scale
  ([`fe594ad`](https://github.com/TensorSpeech/TensorFlowASR/commit/fe594ad3cefe8476c60f7246ad5032209caae55a))

- Buffer size
  ([`4596609`](https://github.com/TensorSpeech/TensorFlowASR/commit/45966096ad8167dcd933f25b2e8cf7df588d97d8))

- Config
  ([`3bbfd77`](https://github.com/TensorSpeech/TensorFlowASR/commit/3bbfd77a2d053df5829774fcc14a69caf648f0c7))

- Config
  ([`6be5428`](https://github.com/TensorSpeech/TensorFlowASR/commit/6be54286d3b34b0fca1a6e18ac30983530db477d))

- Config
  ([`cf435a3`](https://github.com/TensorSpeech/TensorFlowASR/commit/cf435a3539443a62ddba7ec30d408c3563d41526))

- Config
  ([`7611ff8`](https://github.com/TensorSpeech/TensorFlowASR/commit/7611ff814e6b396be360893249a617dddc34c621))

- Config
  ([`6f7f246`](https://github.com/TensorSpeech/TensorFlowASR/commit/6f7f246353656b7cf76f88300463b65929e32e26))

- Config
  ([`c9e4d38`](https://github.com/TensorSpeech/TensorFlowASR/commit/c9e4d38812a3ad9b9a4f2e07720f35eaaff40c49))

- Config
  ([`8268afe`](https://github.com/TensorSpeech/TensorFlowASR/commit/8268afeea0b22b9bbdf5da0d143e9d98ed4f0afb))

- Config
  ([`43d6054`](https://github.com/TensorSpeech/TensorFlowASR/commit/43d60547105eb27e65f09dddbf56144244cdac63))

- Configs
  ([`8bcf0f3`](https://github.com/TensorSpeech/TensorFlowASR/commit/8bcf0f30dd977607678afc96ccf873a3bbcf173c))

- Configs
  ([`2a40da6`](https://github.com/TensorSpeech/TensorFlowASR/commit/2a40da69680bcf8d4487b4e7ccb7edf5d58d9c50))

- Configs
  ([`91a39a2`](https://github.com/TensorSpeech/TensorFlowASR/commit/91a39a24598b6c10109942c70876482bf4d95175))

- Configs
  ([`d541928`](https://github.com/TensorSpeech/TensorFlowASR/commit/d541928ec01fcc3d1a119df9807258c4c83c2228))

- Configs
  ([`111f3ac`](https://github.com/TensorSpeech/TensorFlowASR/commit/111f3aca6e469a940149d4116c701d58d4739b73))

- Configs
  ([`3e88f65`](https://github.com/TensorSpeech/TensorFlowASR/commit/3e88f65682ac26e2380dd3017a9e0b6bd2b0597a))

- Configs
  ([`0556481`](https://github.com/TensorSpeech/TensorFlowASR/commit/055648174e6c1cfa248e32eb429a4ece1c181eb2))

- Fix features source
  ([`1ce3d1c`](https://github.com/TensorSpeech/TensorFlowASR/commit/1ce3d1cdec1511a9da35e5b69bb47cdf5947044d))

- List devices
  ([`5ff3163`](https://github.com/TensorSpeech/TensorFlowASR/commit/5ff3163c43d7c887673b253d6c1e5863ffaee453))

- Logging
  ([`7473458`](https://github.com/TensorSpeech/TensorFlowASR/commit/74734585ccabad56985b938f6edc76cbadd07ecd))

- Remove commented code
  ([`0d75686`](https://github.com/TensorSpeech/TensorFlowASR/commit/0d756863429eaa8e6bc81f34fa3ffc73f6864711))

- Remove commented code
  ([`81a336c`](https://github.com/TensorSpeech/TensorFlowASR/commit/81a336c7278fdf1f529e6228955bd90058359950))

- Security
  ([`66ca1ed`](https://github.com/TensorSpeech/TensorFlowASR/commit/66ca1ed523ba790864906d6751e9cf03225232c8))

- Setup mxp
  ([`56d2afa`](https://github.com/TensorSpeech/TensorFlowASR/commit/56d2afa617335e6b8e7f1f008a17a2a53f5d2f20))

- Setup mxp
  ([`0c8e7c1`](https://github.com/TensorSpeech/TensorFlowASR/commit/0c8e7c1df4cea21df2b422808fc2d9249d2bfbe4))

- Summary
  ([`b70bdd6`](https://github.com/TensorSpeech/TensorFlowASR/commit/b70bdd6d5555bea839f2320f6d17508408628b26))

- Transformer-ctc streaming
  ([`a333bfc`](https://github.com/TensorSpeech/TensorFlowASR/commit/a333bfc5991c2f8f9c80e1d0d5a987f063c3f710))

- Unittest
  ([`af93c2b`](https://github.com/TensorSpeech/TensorFlowASR/commit/af93c2bd5a539ce97fb7c0da7ddd1fc3915d677d))

- Update
  ([`a05494a`](https://github.com/TensorSpeech/TensorFlowASR/commit/a05494a887bc0ac6523b167bd689044ff7cfc408))

- Update configs
  ([`4f77a52`](https://github.com/TensorSpeech/TensorFlowASR/commit/4f77a52edfa1b651fe1bbef0032ec5aaa62bed08))

- Update dataset links
  ([`3104da1`](https://github.com/TensorSpeech/TensorFlowASR/commit/3104da1eb2711efe2b08ed56752aed77389b80c7))

- Update install script
  ([`6338f55`](https://github.com/TensorSpeech/TensorFlowASR/commit/6338f55949cf182713cc8db7af29a19ca8c01dc1))

- Update logging
  ([`4824929`](https://github.com/TensorSpeech/TensorFlowASR/commit/48249296bb728fe0093573fcf928918210ee9ea8))

- Verbose
  ([`31312d5`](https://github.com/TensorSpeech/TensorFlowASR/commit/31312d52198c73dbf654ab6b4d069c5d3f6d1509))

- Vietbud500 256 sentencepiece
  ([`adcaa17`](https://github.com/TensorSpeech/TensorFlowASR/commit/adcaa17836add44e2bbf4cc428a83f0b2742a719))

- Vietbud500 metadata
  ([`3a5b511`](https://github.com/TensorSpeech/TensorFlowASR/commit/3a5b511e422ba121d42bd9a3cdc51ca9bccbc29f))

### Documentation

- Readme
  ([`f092d39`](https://github.com/TensorSpeech/TensorFlowASR/commit/f092d39743d74308cb7959001c4657b98d01e7be))

- Readme
  ([`14124f4`](https://github.com/TensorSpeech/TensorFlowASR/commit/14124f469c481ce6ab4e057f87ca8d49b28978b3))

- Readme
  ([`4a4dece`](https://github.com/TensorSpeech/TensorFlowASR/commit/4a4decef6689ea59d6a070cc2660005f679020fc))

- Update conformer transducer results
  ([`edcb7db`](https://github.com/TensorSpeech/TensorFlowASR/commit/edcb7db49ab8c4ef0a405668775d0c4bec998f11))

- Update conformer transducer/ctc results
  ([`0c19911`](https://github.com/TensorSpeech/TensorFlowASR/commit/0c19911af4d63c3277530f06b67dd31571dd1570))

### Features

- Add kaggle backup and restore callback
  ([`ab83d87`](https://github.com/TensorSpeech/TensorFlowASR/commit/ab83d8700112aa100871dad35dd2a5338d0ae838))

- Add keep checkpoints to small number in callback
  ([`be4a782`](https://github.com/TensorSpeech/TensorFlowASR/commit/be4a782f4c7529db4e8e205898aca07aa390d732))

- Add support for layer norm in conformer conv module
  ([`26c4a5f`](https://github.com/TensorSpeech/TensorFlowASR/commit/26c4a5f08bd19d73d5331c961f016528b47cd6d2))

- Bundle scripts inside package
  ([`9824819`](https://github.com/TensorSpeech/TensorFlowASR/commit/9824819b2d4f7a08fab1ac52af225f3557339097))

- Fix layers, models to tf2.16 with keras 3
  ([`5037896`](https://github.com/TensorSpeech/TensorFlowASR/commit/503789657083110092dadc57c87885ba92470ab2))

- Introduce chunk-wise masking for mha layer
  ([`05b068b`](https://github.com/TensorSpeech/TensorFlowASR/commit/05b068b76164fe69ec5d0812626ef8f4dc2342dd))

- Introduce chunk-wise masking to conformer & transformer
  ([`1edf16a`](https://github.com/TensorSpeech/TensorFlowASR/commit/1edf16a75c98f0320fc73fee68bc14ba5bc63d25))

- Refactor tokenizer, dataset with custom build vocabulary and
  ([`dd17fb6`](https://github.com/TensorSpeech/TensorFlowASR/commit/dd17fb6f17377a371c917b5cd7b958f8d681d5dc))

- Tf2.16 with keras 3
  ([`90cabc2`](https://github.com/TensorSpeech/TensorFlowASR/commit/90cabc242d8ad874d76d2d8032997ce1fd1f5f9a))

- Update models to compatible with keras 3
  ([`a853eba`](https://github.com/TensorSpeech/TensorFlowASR/commit/a853eba3bac704f58e2a383b65b4995933722459))

- **streaming**: Training, inference, gradient accumulation
  ([`232a80c`](https://github.com/TensorSpeech/TensorFlowASR/commit/232a80c7879874b6add283ba461deffea77f6777))


## v2.1.0 (2024-06-09)

### Bug Fixes

- Chars blank
  ([`27a66dd`](https://github.com/TensorSpeech/TensorFlowASR/commit/27a66dd7f74155c4c45aa3bb48e2d197a1314034))

- Configs
  ([`7606d94`](https://github.com/TensorSpeech/TensorFlowASR/commit/7606d94273659564e2565bb8cef94dddf5fa5c7d))

- Replace import tf.keras to keras, update tiny rnnt model result
  ([`63a400c`](https://github.com/TensorSpeech/TensorFlowASR/commit/63a400c76cffdd353ae2fb9f5c100a9970197105))

- Update inference main, tflite docs
  ([`c8d9f06`](https://github.com/TensorSpeech/TensorFlowASR/commit/c8d9f06fa70011a560453c4027bb8b97e9b8d780))

### Chores

- V2.1.0
  ([`2ebd30e`](https://github.com/TensorSpeech/TensorFlowASR/commit/2ebd30e6a527e855b478743bb109574618842493))

### Features

- Register keras custom objects to support load models
  ([`096170a`](https://github.com/TensorSpeech/TensorFlowASR/commit/096170a35e6719d928e88972eeebf5343fc1898d))


## v2.0.1 (2024-05-20)

### Bug Fixes

- Add tfrecords buffer size in bytes for tfrecords dataset
  ([`316694f`](https://github.com/TensorSpeech/TensorFlowASR/commit/316694f0590815a3626fe0131c56a001e45876af))

- Tflite conversion and inference
  ([`a4d411d`](https://github.com/TensorSpeech/TensorFlowASR/commit/a4d411dab1211754fd8d8372a2616f4bb1196304))

- Update contextnet results
  ([`7d18d23`](https://github.com/TensorSpeech/TensorFlowASR/commit/7d18d2327d65caf8b234b903d42ee22d72039078))

### Chores

- Add pre-commit
  ([`0635cfe`](https://github.com/TensorSpeech/TensorFlowASR/commit/0635cfe4593a6113902a6cb3ead97988e84d9bf5))

- Add tfrecords buffer size to config
  ([`40b5072`](https://github.com/TensorSpeech/TensorFlowASR/commit/40b50728dd35ddb5f4889c43a8fcda7f3a4d1d21))

- Create wiki-publish.yml
  ([`3cf575f`](https://github.com/TensorSpeech/TensorFlowASR/commit/3cf575f8fdf343bcad8345a2ea6da09344c9c589))

- Update docs
  ([`ec3ccc2`](https://github.com/TensorSpeech/TensorFlowASR/commit/ec3ccc2b9713904f68645ef48b81eb3407255b64))

- Update docs
  ([`fe33a68`](https://github.com/TensorSpeech/TensorFlowASR/commit/fe33a68324671528b31d1bd9f20ad335316593e1))

- Update links and tutorials
  ([`e915fcd`](https://github.com/TensorSpeech/TensorFlowASR/commit/e915fcd06333920b5d4b656c1fb1cc73b96d83be))

- Update readme links
  ([`7480128`](https://github.com/TensorSpeech/TensorFlowASR/commit/74801289dadf6200c9277888e11e14dd8d8c709f))

- Update wiki-publish.yml
  ([`36b070b`](https://github.com/TensorSpeech/TensorFlowASR/commit/36b070b981e499388f6ec61fd5ecca8325aa7c48))

- Update wiki-publish.yml
  ([`565cf82`](https://github.com/TensorSpeech/TensorFlowASR/commit/565cf82eb45628338feaa771ed9defad575d85c9))

- Update wiki-publish.yml
  ([`31e17bc`](https://github.com/TensorSpeech/TensorFlowASR/commit/31e17bc40a36b2ce5836508531d590e152dec9a9))

- Update wiki-publish.yml
  ([`5f1c416`](https://github.com/TensorSpeech/TensorFlowASR/commit/5f1c416c45429746ec91097ec0b517b318fa628d))

- Update wiki-publish.yml
  ([`d322f34`](https://github.com/TensorSpeech/TensorFlowASR/commit/d322f34e1d93718e9015d4d1e3bfd6024381eed0))

- Update wiki-publish.yml
  ([`2883203`](https://github.com/TensorSpeech/TensorFlowASR/commit/2883203d6bcde9b5d1833c0e06c1d47222527891))

- Update wiki-publish.yml
  ([`0cdb64c`](https://github.com/TensorSpeech/TensorFlowASR/commit/0cdb64c775281f2a2d2803636de52e8a3e439eb1))

- Update wiki-publish.yml
  ([`b1408a6`](https://github.com/TensorSpeech/TensorFlowASR/commit/b1408a6b25a17108d193597eaa553ec4d0670e11))

- Update wiki-publish.yml
  ([`4658bf9`](https://github.com/TensorSpeech/TensorFlowASR/commit/4658bf95d0a28b634f9fc5ad3cdbff0ccf794973))

- Update wiki-publish.yml
  ([`7aeedf6`](https://github.com/TensorSpeech/TensorFlowASR/commit/7aeedf69513625cf24c8948e92bf1e489b586f5f))

- Update wiki-publish.yml
  ([`38bd08e`](https://github.com/TensorSpeech/TensorFlowASR/commit/38bd08ee187de3010cd92f7e38d4edadbee91d45))


## v2.0.0 (2024-05-05)

### Bug Fixes

- Accumulate
  ([`45d4d0e`](https://github.com/TensorSpeech/TensorFlowASR/commit/45d4d0e93afec5a4add17ca5f8115272b64ac624))

- Add auto mask model options
  ([`1539e13`](https://github.com/TensorSpeech/TensorFlowASR/commit/1539e137a6d27909fd2d33f5a38ec0a145ce0e49))

- Add bn type for ds2
  ([`9031fc8`](https://github.com/TensorSpeech/TensorFlowASR/commit/9031fc8d158f574f48a4fc5d59b9e160bc73ff59))

- Add callback and concat results
  ([`4a79253`](https://github.com/TensorSpeech/TensorFlowASR/commit/4a792531b454ea536029884a4ff5034ffd27b113))

- Add count in results of step functions
  ([`efc470a`](https://github.com/TensorSpeech/TensorFlowASR/commit/efc470a3472d0c6c0bddb65ca7081e0999b56009))

- Add default repodir in j2 rendering
  ([`e4b9167`](https://github.com/TensorSpeech/TensorFlowASR/commit/e4b9167dc4b9b1c9fce95adf368a2a76dcc5776f))

- Add dense as pointwise conv
  ([`291cb96`](https://github.com/TensorSpeech/TensorFlowASR/commit/291cb9600fe636de3cbc4c78afeee23c052824db))

- Add dynamic relpe option
  ([`dab6c7b`](https://github.com/TensorSpeech/TensorFlowASR/commit/dab6c7bd4a70fed74777661709cb051b3adf5704))

- Add early stopping
  ([`be6772a`](https://github.com/TensorSpeech/TensorFlowASR/commit/be6772a0d9812612bb8bd63d2912d5f8ef04abbe))

- Add gwn
  ([`875f384`](https://github.com/TensorSpeech/TensorFlowASR/commit/875f384048f9787f478e638b5736304aed13f274))

- Add gwn to ctc models
  ([`7e6af0e`](https://github.com/TensorSpeech/TensorFlowASR/commit/7e6af0e09f36139eaf32d2949634da7412f79e5f))

- Add ignore recompute metadata
  ([`0567eb0`](https://github.com/TensorSpeech/TensorFlowASR/commit/0567eb074581988b68524a00f3e593fc0026ab54))

- Add initial lr in transformer scheduler
  ([`3f3933a`](https://github.com/TensorSpeech/TensorFlowASR/commit/3f3933af06fc1aac8c7946c8bb99c260804e130d))

- Add initializer ds2
  ([`9aa9c73`](https://github.com/TensorSpeech/TensorFlowASR/commit/9aa9c735aa7753bf339b36e049b79ad7a23c04aa))

- Add instruction for apple sillicon installation for tensorflow-text
  ([`be3d4ed`](https://github.com/TensorSpeech/TensorFlowASR/commit/be3d4ed5ca69f8e7d5fd6afefe4ae82084bf0fa2))

- Add length_as_output
  ([`e9ec388`](https://github.com/TensorSpeech/TensorFlowASR/commit/e9ec38849982866e6e587ff77060efb51fa4bb81))

- Add loss scale by grad accumulation
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Add masking for relpos
  ([`ee07fbe`](https://github.com/TensorSpeech/TensorFlowASR/commit/ee07fbe9677d3180d5b40f3918bf42e034d22204))

- Add memory length to transformer
  ([`1b74106`](https://github.com/TensorSpeech/TensorFlowASR/commit/1b7410686e734eabd0c5a0f5aee44fde558f246f))

- Add missing cast warprnnt loss
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Add missing watch
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Add name
  ([`37cb0f8`](https://github.com/TensorSpeech/TensorFlowASR/commit/37cb0f8afb1ac26224a0dfcb1f2b693d39ca627d))

- Add norm and scale options for conformer
  ([`776aa5e`](https://github.com/TensorSpeech/TensorFlowASR/commit/776aa5ef8d16bb17d2195b1296d0f98b762084bc))

- Add option character_coverage
  ([`bd6f737`](https://github.com/TensorSpeech/TensorFlowASR/commit/bd6f737f44b10b1b8c74b84a4248af357dedaf0a))

- Add option jit_compile for training
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add option to mask upper triangle in rel shift
  ([`7f0a171`](https://github.com/TensorSpeech/TensorFlowASR/commit/7f0a17114012c1205f526654597475f6f5decba5))

- Add prediction rnn unroll
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add relpe direction
  ([`cdfe56a`](https://github.com/TensorSpeech/TensorFlowASR/commit/cdfe56a1047ad805b8025f304bf042c83eb749ab))

- Add reset states to eval steps
  ([`693bb4c`](https://github.com/TensorSpeech/TensorFlowASR/commit/693bb4cb9af3cfe9b918e681a648f37ce3f4ebab))

- Add reset states to reset metrics
  ([`b85c921`](https://github.com/TensorSpeech/TensorFlowASR/commit/b85c9210ca9b77b67219cb758f2d18682766a47f))

- Add soft device placement for compute wer,cer on validation
  ([`32d2a43`](https://github.com/TensorSpeech/TensorFlowASR/commit/32d2a43a250e13118f17098e4d15dbfac64c06c3))

- Add stop gradient on weight noise
  ([`5a37cab`](https://github.com/TensorSpeech/TensorFlowASR/commit/5a37cab7c38a8e4cfc8017e5abeecd00731b4bf4))

- Add strict_auto mxp
  ([`f034802`](https://github.com/TensorSpeech/TensorFlowASR/commit/f034802b07f0050ea08da6ed4a84dbd4dc61cc9e))

- Add subsampling dropout
  ([`a6410c8`](https://github.com/TensorSpeech/TensorFlowASR/commit/a6410c8cfb7e630a3f185d1e6ffb24a7f7ee903d))

- Add support for tf extract text featurizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add support ga for contextnet
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add temporal dense
  ([`5fa7ef9`](https://github.com/TensorSpeech/TensorFlowASR/commit/5fa7ef93167c2c0bc5be6086a9e0391fa21c3b1f))

- Add tiny conformer
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Add wordpiece for contextnet
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add zero_output_for_mask=True for rnn
  ([`ac1ab54`](https://github.com/TensorSpeech/TensorFlowASR/commit/ac1ab5480d608c55f8d7b80dd76ffb986003a55f))

- Allow force cpu loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Apply gwn
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Apply tf.function to run step
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Asr dataset
  ([`b36d81e`](https://github.com/TensorSpeech/TensorFlowASR/commit/b36d81e5ad78b112cabd668313e5a7b1fb57ff0a))

- Asr dataset
  ([`fa40248`](https://github.com/TensorSpeech/TensorFlowASR/commit/fa40248050ba13dd45d59861ae1c604c7d357705))

- Attention mask
  ([`2f63e26`](https://github.com/TensorSpeech/TensorFlowASR/commit/2f63e26d933965912716b70a20feb3cc29ab1866))

- Augment and conf
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Auto mxp
  ([`c7fb0d4`](https://github.com/TensorSpeech/TensorFlowASR/commit/c7fb0d427c31f8ad76db6c9e1d81b562cf526935))

- Avoid using mask in softmax
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Base transducer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Blank of character featurizer
  ([`2f58527`](https://github.com/TensorSpeech/TensorFlowASR/commit/2f58527eb9dde94b3850c9adf9fc5400da09a8f0))

- Build from corpus
  ([`4ebfc58`](https://github.com/TensorSpeech/TensorFlowASR/commit/4ebfc58b2a7ff37983e54ce37b22d3c877cc1491))

- Cache conf
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Casting position encoding
  ([`c987879`](https://github.com/TensorSpeech/TensorFlowASR/commit/c98787944ad123a001167e68f67bd121d71f4c49))

- Casting transformer scheduler
  ([`4669eee`](https://github.com/TensorSpeech/TensorFlowASR/commit/4669eee570cf390205961e8ea01bdc2c39bde89d))

- Change default mask value
  ([`5e2092d`](https://github.com/TensorSpeech/TensorFlowASR/commit/5e2092d8d9afa4d58eb811a0d5e9f8f163320558))

- Change order in rnnt block
  ([`391e0e8`](https://github.com/TensorSpeech/TensorFlowASR/commit/391e0e8996e9e6a71abd18c778a1fb96fdd2e75c))

- Conf
  ([`de2f237`](https://github.com/TensorSpeech/TensorFlowASR/commit/de2f23767574f61ca0b40dc0590dd0822d4b6b73))

- Conf
  ([`db2bf34`](https://github.com/TensorSpeech/TensorFlowASR/commit/db2bf34479e7fff6e736ca1dcbbe90febf8ada02))

- Config and ga loss calc
  ([`8c255a3`](https://github.com/TensorSpeech/TensorFlowASR/commit/8c255a3a7fc69c7197fc65e677443f69f0f59f44))

- Config template
  ([`09352b1`](https://github.com/TensorSpeech/TensorFlowASR/commit/09352b10a8c65aacd9d2c7717beeb313c1465c96))

- Config template
  ([`793781e`](https://github.com/TensorSpeech/TensorFlowASR/commit/793781ee4710b86322cc7b1d31b292121e059119))

- Configs
  ([`ed15031`](https://github.com/TensorSpeech/TensorFlowASR/commit/ed1503173a65f9b700bcf98c0abcdc0422044f40))

- Configs
  ([`0c6574d`](https://github.com/TensorSpeech/TensorFlowASR/commit/0c6574d48205d8837aee17fb4bdac914dc17db73))

- Conformer architecture
  ([`07aecbb`](https://github.com/TensorSpeech/TensorFlowASR/commit/07aecbb8c8130246064ae234af91fd01df3872d8))

- Conformer config
  ([`bfab670`](https://github.com/TensorSpeech/TensorFlowASR/commit/bfab670c4cc5f36e8467d00495ded357f30b6c54))

- Conformer conv module
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Conformer ctc
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Conformer ctc
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Conformer encoder
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Contextnet conf
  ([`6d729aa`](https://github.com/TensorSpeech/TensorFlowASR/commit/6d729aa1f8487b9a7a23deac750feca74783ef2d))

- Contextnet conf
  ([`4bd5355`](https://github.com/TensorSpeech/TensorFlowASR/commit/4bd5355a58993d095fa4ce80dafd13060e70bae6))

- Contextnet dmodel
  ([`9f27a10`](https://github.com/TensorSpeech/TensorFlowASR/commit/9f27a10cd312f425ba76f293376140142bf1e1f8))

- Conv2d subsampling with norm
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Correct global batch size calc
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Correct relative positional encoding, rel left shift in relmha
  ([`7a8dd3b`](https://github.com/TensorSpeech/TensorFlowASR/commit/7a8dd3b5c62414da2a83253b552cb6782e16e2c3))

- Correct relmhsa module in conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Correct tape watch
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Create tfrecords
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Ctc beam search recognize
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Ctc loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Ctc reduced length
  ([`0240a38`](https://github.com/TensorSpeech/TensorFlowASR/commit/0240a3879eef14b4606d1b2ed31848b7c9cba2b0))

- Customized softmax
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Data config
  ([`e04e4b3`](https://github.com/TensorSpeech/TensorFlowASR/commit/e04e4b362d9561a2f5ccbed42ed1ff47172930eb))

- Dataset
  ([`9228877`](https://github.com/TensorSpeech/TensorFlowASR/commit/9228877beb05d966fc9970535b17b61a2642cdae))

- Dataset map deterministic
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Default norm false
  ([`652bb92`](https://github.com/TensorSpeech/TensorFlowASR/commit/652bb92864d328ac4f1d72430a8953247846c8f2))

- Default not use cpu loss
  ([`8f76d77`](https://github.com/TensorSpeech/TensorFlowASR/commit/8f76d77899fdefb6f74f73cf44e116a5b3f9ef37))

- Deps
  ([`a85ce85`](https://github.com/TensorSpeech/TensorFlowASR/commit/a85ce851f223b890c7ebcf6efad2ffdbf79c5e30))

- Deps tf 2.13 gpu
  ([`941f0f3`](https://github.com/TensorSpeech/TensorFlowASR/commit/941f0f3448144c879e7657cb7ced56074894c0a3))

- Deps tf 2.14 gpu
  ([`391d6d4`](https://github.com/TensorSpeech/TensorFlowASR/commit/391d6d42c4f1e0818477ee28e3461fc83f1b5f9c))

- Disable eager
  ([`55227df`](https://github.com/TensorSpeech/TensorFlowASR/commit/55227df94e427ecf420a17e00b0ae04e76d2a2e6))

- Disable loss scale
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Drop support for tf < 2.8 + update asr dataset
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Ds2
  ([`7a3f6d7`](https://github.com/TensorSpeech/TensorFlowASR/commit/7a3f6d757efedc420cb233bccb402192df322bf0))

- Ds2
  ([`a80cc50`](https://github.com/TensorSpeech/TensorFlowASR/commit/a80cc506998464c7838052f1e0c5fbbdb69c0d31))

- Ds2 bn only after convs, update conf
  ([`e0ee0bf`](https://github.com/TensorSpeech/TensorFlowASR/commit/e0ee0bf87af41178e2c374564f3e1afbd25d06fc))

- Ds2 conf
  ([`12109cc`](https://github.com/TensorSpeech/TensorFlowASR/commit/12109ccfb07b3772d34be24931913c5947cf0d67))

- Ds2 conf
  ([`b6965b3`](https://github.com/TensorSpeech/TensorFlowASR/commit/b6965b3b9cc116d609500fcdf2feef66d546b554))

- Ds2 conf
  ([`23db6af`](https://github.com/TensorSpeech/TensorFlowASR/commit/23db6af23480a7f3398587972506e276781ad20d))

- Ds2 conf
  ([`7444dfc`](https://github.com/TensorSpeech/TensorFlowASR/commit/7444dfca8a4f34de8559d96f8887e126e24ce34c))

- Ds2 model and conf
  ([`640dcd4`](https://github.com/TensorSpeech/TensorFlowASR/commit/640dcd497ecd354204a0899febe7e5c61203c0fb))

- Ds2 model and conf
  ([`bb1bd97`](https://github.com/TensorSpeech/TensorFlowASR/commit/bb1bd970d76c47deae4e9e57e66a7ef853c20294))

- Ds2 rowconv
  ([`635199f`](https://github.com/TensorSpeech/TensorFlowASR/commit/635199fbd203d70fad7ae3e993bdf16919fe826c))

- Ds2, rnnt, contextnet
  ([`a07eb66`](https://github.com/TensorSpeech/TensorFlowASR/commit/a07eb66ce43c5f94b0113416e247e3607c0509cf))

- Dtype
  ([`e9a37be`](https://github.com/TensorSpeech/TensorFlowASR/commit/e9a37be2044548cb227815f2f45f569fb538acd3))

- Dtype mha
  ([`f735b9c`](https://github.com/TensorSpeech/TensorFlowASR/commit/f735b9c76d4a9379f8f95aca2539481f006acc11))

- Dtype of blurpool
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Dtype of rnn layers
  ([`0a9ca7e`](https://github.com/TensorSpeech/TensorFlowASR/commit/0a9ca7e51f4dd8815ca0c9e730cc2e207c727bc0))

- Env util mxp
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Env utils
  ([`3c0fa11`](https://github.com/TensorSpeech/TensorFlowASR/commit/3c0fa1104ebd70fa2241622422fcb7a576240f6b))

- Example configs
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Fix pylint warnings
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Float32 in layernorm for numeric stability
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Force float32 for speech extraction layer
  ([`70e2650`](https://github.com/TensorSpeech/TensorFlowASR/commit/70e2650ec8dadf27e4112aaa79e7f204064bcdd9))

- Formatting
  ([`fd0c374`](https://github.com/TensorSpeech/TensorFlowASR/commit/fd0c37418d4d1d59fe92b988b9ccb2edd07294cd))

- Gaussian weight noise
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- General example scripts
  ([`36fda64`](https://github.com/TensorSpeech/TensorFlowASR/commit/36fda64651ea9ace4b95fdfdc1bcb6711c11ba73))

- Gradient
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Gradient accumulation
  ([`5744fd3`](https://github.com/TensorSpeech/TensorFlowASR/commit/5744fd3bc9886ba8e26b56c8077c0d55b88dcbc5))

- Gradient tape
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Gradient tape
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Greedy transducer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Header output test tsv
  ([`abdcfb0`](https://github.com/TensorSpeech/TensorFlowASR/commit/abdcfb0246627cdbce176d651cc52f6ea366c0ce))

- Import orders + update base summary
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Imports
  ([`44545de`](https://github.com/TensorSpeech/TensorFlowASR/commit/44545de697aeb6c42c49b3add86b8691d333cfbb))

- Incorrect tensorflow-io versions
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Init scope
  ([`44d2b25`](https://github.com/TensorSpeech/TensorFlowASR/commit/44d2b25fd3db37cad731c61b32908f729fa528aa))

- Install rnnt loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Interleave relpe
  ([`e4c7c2f`](https://github.com/TensorSpeech/TensorFlowASR/commit/e4c7c2fe2e8bdeb6a76c3b3d4c3a8e0af4e11f69))

- Iterator
  ([`8c41e3b`](https://github.com/TensorSpeech/TensorFlowASR/commit/8c41e3b2c71ce2c16166466d381cd44695316216))

- Iterator
  ([`3f3744c`](https://github.com/TensorSpeech/TensorFlowASR/commit/3f3744c19aadb4cbac002173da08744848c06b00))

- Limit float32 for mxp
  ([`0d3fa27`](https://github.com/TensorSpeech/TensorFlowASR/commit/0d3fa273032be0bd0aa1115996b5a144ac4ff4d9))

- Logarithm of features, augmentations, conf
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Logging and add sentencepiece 256
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Loss
  ([`ada358b`](https://github.com/TensorSpeech/TensorFlowASR/commit/ada358b37f34b2fb44fcf702c5d99aa16b6dd5b6))

- Lr
  ([`d599e30`](https://github.com/TensorSpeech/TensorFlowASR/commit/d599e30a7565ddfceb4e64d6a8fe81bdb6fa4dcd))

- Lr schedules
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Mask and output encoder length
  ([`1ca34c7`](https://github.com/TensorSpeech/TensorFlowASR/commit/1ca34c7252c3448422bb0c98208bfd2fcf3584d4))

- Mask position encoding
  ([`e3ee002`](https://github.com/TensorSpeech/TensorFlowASR/commit/e3ee002289bc74625769d0f5b1ba2976ad81fb7d))

- Mask position encoding
  ([`9690269`](https://github.com/TensorSpeech/TensorFlowASR/commit/9690269b7560959fe96144e10e42ab22ab82f476))

- Masked fill
  ([`303b4c5`](https://github.com/TensorSpeech/TensorFlowASR/commit/303b4c597b542d15500a5a030e413a8bc3ed0f69))

- Masking computation with original mask keeping
  ([`787c11a`](https://github.com/TensorSpeech/TensorFlowASR/commit/787c11a0d881c62424cc76ec30ea5010f00a3855))

- Masking sample weight
  ([`1620471`](https://github.com/TensorSpeech/TensorFlowASR/commit/1620471ed642ab1f2fade6dd55db1d96dd8bba57))

- Memory variables
  ([`69b0e40`](https://github.com/TensorSpeech/TensorFlowASR/commit/69b0e4069353ba86cdbc4dab26c42d226b12cdfb))

- Metrics for ga
  ([`5d41f0b`](https://github.com/TensorSpeech/TensorFlowASR/commit/5d41f0be3e721866b6d93031066ab2c63714e65f))

- Mha
  ([`070d6e1`](https://github.com/TensorSpeech/TensorFlowASR/commit/070d6e1d3d0c9e511f4e03bb85e425995337795c))

- Mha
  ([`5f2d1e1`](https://github.com/TensorSpeech/TensorFlowASR/commit/5f2d1e196f9a40f7bcc48526a76affd371c8b5ba))

- Mha attention mask
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Mha dtype
  ([`083b82a`](https://github.com/TensorSpeech/TensorFlowASR/commit/083b82ad1066670eae7e1cd05f89d8122af5ea24))

- Mha maked softmax
  ([`0c55aab`](https://github.com/TensorSpeech/TensorFlowASR/commit/0c55aabebd2f4f78ed7730729777d0e7661edcb8))

- Mha mask
  ([`1061ff4`](https://github.com/TensorSpeech/TensorFlowASR/commit/1061ff4a45a749e5c9fdccdfcb5933211d576397))

- Mha mask
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Mha rel shift
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Mha rel shift and mask
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Mhsa imports
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Missing dtype assignments
  ([`db0a931`](https://github.com/TensorSpeech/TensorFlowASR/commit/db0a931f21854385ad1af780c590745629418429))

- Models do not use caching
  ([`91ff881`](https://github.com/TensorSpeech/TensorFlowASR/commit/91ff8816b41507a2e25d718a37809541d61a5c10))

- Multihead attention conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Multihead relative attention
  ([`ec75012`](https://github.com/TensorSpeech/TensorFlowASR/commit/ec750128232df200f67e6f546f9d00f83683ec6a))

- Multihead relative attention
  ([`874d2b2`](https://github.com/TensorSpeech/TensorFlowASR/commit/874d2b298ff51764751369f36bac34602834854c))

- Multihead relative attention scores computation
  ([`f75dd41`](https://github.com/TensorSpeech/TensorFlowASR/commit/f75dd41fa550fb549dd727159e8950273ca190c3))

- Multihead relative attention, add conformer ctc
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Mxp assignment in compile
  ([`cb00f92`](https://github.com/TensorSpeech/TensorFlowASR/commit/cb00f92394ad025f037847a9b17a1c691603baca))

- Naive rnnt loss
  ([`cf06051`](https://github.com/TensorSpeech/TensorFlowASR/commit/cf06051c4c9cf4d5895f394becbf5674d878afc6))

- Naive rnnt loss
  ([`7b9a30a`](https://github.com/TensorSpeech/TensorFlowASR/commit/7b9a30a923cc5f7cdf759f6d7ffdc2d28b5e3798))

- New config for conformer encoder
  ([`d22a4c8`](https://github.com/TensorSpeech/TensorFlowASR/commit/d22a4c8cb78ff904f3365785971f0edddacd5562))

- Num class wp features
  ([`8422742`](https://github.com/TensorSpeech/TensorFlowASR/commit/8422742d7301535ca8b918d49c2e2a76b3a1ade2))

- Numerical stability
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- One hot blank layer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Optimize character featurizer
  ([`3e8b8c9`](https://github.com/TensorSpeech/TensorFlowASR/commit/3e8b8c929a9ec42d29855217052b7758a5cd67a7))

- Padding and masking ds2
  ([`df5c4ba`](https://github.com/TensorSpeech/TensorFlowASR/commit/df5c4babd524d20db05580bed3617aecf2af3889))

- Pe
  ([`034c4be`](https://github.com/TensorSpeech/TensorFlowASR/commit/034c4be49d6691bef3fc5f0b762f07485f5d8e61))

- Pe
  ([`d28a8d0`](https://github.com/TensorSpeech/TensorFlowASR/commit/d28a8d0e9a99bbba2594cf190ac22ba0256266a8))

- Pe stop gradient
  ([`4424cda`](https://github.com/TensorSpeech/TensorFlowASR/commit/4424cda3b6625ea779d0e34332f57bbb005ce85d))

- Per replica model's outputs shape
  ([`00b5774`](https://github.com/TensorSpeech/TensorFlowASR/commit/00b5774853f3ed2ba2ab1d4a8cf3c6e427d41b25))

- Predict step
  ([`a7ba85c`](https://github.com/TensorSpeech/TensorFlowASR/commit/a7ba85cae510ea3ca81888a7e8bd59249fe7dafd))

- Prediction
  ([`764878c`](https://github.com/TensorSpeech/TensorFlowASR/commit/764878ca11740d19ea1c211fd1227abf2700c0ba))

- Prepare metadata script
  ([`8afd3f6`](https://github.com/TensorSpeech/TensorFlowASR/commit/8afd3f670188310c949aacfec81b3a8935f9ddc9))

- Readme installation
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Reduction
  ([`7124019`](https://github.com/TensorSpeech/TensorFlowASR/commit/71240196605deec2836a501b764610a7f28ea677))

- Refactor
  ([`bdfcd55`](https://github.com/TensorSpeech/TensorFlowASR/commit/bdfcd552e4887dec73560f1567db1c8ac39ffc2c))

- Refactor
  ([`293b516`](https://github.com/TensorSpeech/TensorFlowASR/commit/293b516d3057ad32f37c4dff575ebd851a1885e4))

- Refactor code, add pylint
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Regularizers of batch and layer norm, force float32 for rnn in mixed_bfloat16
  ([`94bf220`](https://github.com/TensorSpeech/TensorFlowASR/commit/94bf22088a47486d9cbca9edae71ae24b19a944a))

- Rel shift
  ([`039982e`](https://github.com/TensorSpeech/TensorFlowASR/commit/039982e2c921869468ab45105dded0ccd23998bf))

- Relation multihead attention conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Relative pos encoding conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Relpe
  ([`5324f21`](https://github.com/TensorSpeech/TensorFlowASR/commit/5324f2155da62cf13eef9112d1dea19f29cfba92))

- Remove batch loss
  ([`e15c609`](https://github.com/TensorSpeech/TensorFlowASR/commit/e15c60990e73c98467fd4d30f4186220c24846a9))

- Remove dense_as_pointwise and add bn sync
  ([`0db4662`](https://github.com/TensorSpeech/TensorFlowASR/commit/0db46625b5e6f1a3f4a17c80cec49bf8fc1fca31))

- Remove experimental
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Remove loss scale by grad accumulation
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Remove pillow from requirements
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Remove predict in validation stage
  ([`677e590`](https://github.com/TensorSpeech/TensorFlowASR/commit/677e5901ed410c900dacf77e724a5a5d4c079d24))

- Remove tensorflow datasets dependency
  ([`7dc96b1`](https://github.com/TensorSpeech/TensorFlowASR/commit/7dc96b193f15f6a1da6da3808ae21c423eb3f00b))

- Remove unnessessary tape.watch
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Requirements
  ([`569c5d0`](https://github.com/TensorSpeech/TensorFlowASR/commit/569c5d022bf4147d4313c69183b6ed963f9e1df0))

- Requirements
  ([`ea1363b`](https://github.com/TensorSpeech/TensorFlowASR/commit/ea1363bb37b7e0832f6fca58d27a07f41d7f60ee))

- Restructure
  ([`f37b153`](https://github.com/TensorSpeech/TensorFlowASR/commit/f37b153eb06778f6a80415def2c4b49aad7ddbca))

- Restructure and update global shapes
  ([`8f76c14`](https://github.com/TensorSpeech/TensorFlowASR/commit/8f76c1458b00127f39f33f8040c593a2e5e77840))

- Restructure and update global shapes
  ([`48c6e7f`](https://github.com/TensorSpeech/TensorFlowASR/commit/48c6e7f14028b095417f51fcb5411444955d1181))

- Rnn layer not working with bfloat16 for tf < 2.13
  ([`7e205a9`](https://github.com/TensorSpeech/TensorFlowASR/commit/7e205a971d4381c6953c61272ab354a7662d8b56))

- Rnn transducer
  ([`a6f81ab`](https://github.com/TensorSpeech/TensorFlowASR/commit/a6f81ab427d3a40348c92c181cf3922660211468))

- Rnnt loss
  ([`f0e7622`](https://github.com/TensorSpeech/TensorFlowASR/commit/f0e7622644ce21c80b62dade1fa37d24fb29293a))

- Scripts
  ([`fb12738`](https://github.com/TensorSpeech/TensorFlowASR/commit/fb1273851ceb34a120b121261e71adfe69b68d8f))

- Scripts
  ([`424c21e`](https://github.com/TensorSpeech/TensorFlowASR/commit/424c21e45ee51f31e3f555f9745ce16240b7e306))

- Sentencepiece
  ([`d3a1d6f`](https://github.com/TensorSpeech/TensorFlowASR/commit/d3a1d6ff6e615e8a21da6758a547f31ff999fa27))

- Sentencepiece does not allow to have single whitespace token
  ([`eda7ff0`](https://github.com/TensorSpeech/TensorFlowASR/commit/eda7ff0be30544793a693d8b7d8b9b248d42bac9))

- Setup devices
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Setup mxp
  ([`6e9c985`](https://github.com/TensorSpeech/TensorFlowASR/commit/6e9c98534ff47382c94a98d25605721b5e64a557))

- Shape rnnt loss
  ([`4a12235`](https://github.com/TensorSpeech/TensorFlowASR/commit/4a1223598a93bf903a3af1bb1aa83490475b8150))

- Slice batch
  ([`696fe75`](https://github.com/TensorSpeech/TensorFlowASR/commit/696fe75c681708b58ba3fc82cabfd1465950807b))

- Small update default value for conformer
  ([`bef5048`](https://github.com/TensorSpeech/TensorFlowASR/commit/bef5048fe877ca6e93463c97f51222990b27e02a))

- Sp 1k librispeech
  ([`fc7631a`](https://github.com/TensorSpeech/TensorFlowASR/commit/fc7631afb0aded31ef0daf8736d862c2a87ef7f6))

- Sp 256 librispeech
  ([`1631c09`](https://github.com/TensorSpeech/TensorFlowASR/commit/1631c0996ae60c6b79a0c244235de826f6b5d93f))

- Sp num threads
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Sp whitespace
  ([`d25b675`](https://github.com/TensorSpeech/TensorFlowASR/commit/d25b67518ab37886573d7da53146b480880a5a1a))

- Sp whitespace
  ([`9bab1ca`](https://github.com/TensorSpeech/TensorFlowASR/commit/9bab1ca6acc97916823e6f0704e252e6b198ee31))

- Sp whitespace
  ([`becd557`](https://github.com/TensorSpeech/TensorFlowASR/commit/becd55781dffb300329e210b76447408316afe20))

- Speech feature extraction
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Split train inputs
  ([`481d953`](https://github.com/TensorSpeech/TensorFlowASR/commit/481d953db057a7b7c1ba796894e4801a8aa493f7))

- Stable mha
  ([`b4c04c5`](https://github.com/TensorSpeech/TensorFlowASR/commit/b4c04c5877a509e524cbd82cf0731f98a316954a))

- Stop gradient gauss weight noise
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Structure
  ([`a3e5d20`](https://github.com/TensorSpeech/TensorFlowASR/commit/a3e5d200c4df0f08d9f6229e43fa52ecce3b61f7))

- Subsampling
  ([`e803026`](https://github.com/TensorSpeech/TensorFlowASR/commit/e8030269f8f905ef864c74cc9aaab085fb1c335b))

- Subsampling layer with masking
  ([`4c70b11`](https://github.com/TensorSpeech/TensorFlowASR/commit/4c70b111f3565684e67673819ffb3f55cd1b7304))

- Tape watch
  ([`e3a7bdf`](https://github.com/TensorSpeech/TensorFlowASR/commit/e3a7bdfffac58995b44b5ed08fc8367c22ce0e5e))

- Tape watch
  ([`ea985dc`](https://github.com/TensorSpeech/TensorFlowASR/commit/ea985dca18f74993cc01cff602f0bf179cd6a24e))

- Test mask
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Test results
  ([`4d41027`](https://github.com/TensorSpeech/TensorFlowASR/commit/4d41027e6dd6eb1bd67733b5ea2da9013d09623b))

- Test view model
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Tf range
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Tflite export
  ([`bebce3c`](https://github.com/TensorSpeech/TensorFlowASR/commit/bebce3c331b9abce92d9846556352e4b609fe606))

- Time reduction factor
  ([`4f26bc9`](https://github.com/TensorSpeech/TensorFlowASR/commit/4f26bc97286811a834e77e35eb8382eba00c7b7f))

- Tpu strategy
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Train scripts
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Transducer decoder gauss weight noise
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Transducer decoding for single item batch
  ([`abce732`](https://github.com/TensorSpeech/TensorFlowASR/commit/abce732e713a6be013007c3744fbb4923bc412df))

- Transformer encoder ctc
  ([`629506f`](https://github.com/TensorSpeech/TensorFlowASR/commit/629506f6b9ad7b19eb905f3411835bc8da45fc04))

- Transformer memory length
  ([`68b0c6e`](https://github.com/TensorSpeech/TensorFlowASR/commit/68b0c6eae970316d281266e35b37b36192a7ccbf))

- Unroll for bfloat16
  ([`2a8cf86`](https://github.com/TensorSpeech/TensorFlowASR/commit/2a8cf8649edb73cfc1aa806f264ee942c212c56e))

- Unuse loss scale for tpu
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Upate conf
  ([`56bb01a`](https://github.com/TensorSpeech/TensorFlowASR/commit/56bb01a89c8a1e25671f54efcd2b7ad9217c496d))

- Update
  ([`94cad4e`](https://github.com/TensorSpeech/TensorFlowASR/commit/94cad4ee8eb868f5983d4e5050dc73c919c0e0f2))

- Update
  ([`37293eb`](https://github.com/TensorSpeech/TensorFlowASR/commit/37293eb60451e78b2e9ae148357f466f7a148614))

- Update
  ([`2a2c8b4`](https://github.com/TensorSpeech/TensorFlowASR/commit/2a2c8b498bb15a2e2dec02623d32b89377803ec0))

- Update
  ([`a0f5ebb`](https://github.com/TensorSpeech/TensorFlowASR/commit/a0f5ebb6a4b90d37e39dd7ba566cdb9a5342385d))

- Update
  ([`e7107fe`](https://github.com/TensorSpeech/TensorFlowASR/commit/e7107fec6849a11bb8a2cd0ceb20fdbcdfe375ee))

- Update
  ([`9a1c3cd`](https://github.com/TensorSpeech/TensorFlowASR/commit/9a1c3cddc680d4f5e9a1973769d7aae72d8d286d))

- Update
  ([`3249e4c`](https://github.com/TensorSpeech/TensorFlowASR/commit/3249e4c7796709d08f1a67c620b07a48e182f927))

- Update
  ([`1994958`](https://github.com/TensorSpeech/TensorFlowASR/commit/1994958a782c5c2aa6c07dcec6869bc360c25c8e))

- Update
  ([`74b8462`](https://github.com/TensorSpeech/TensorFlowASR/commit/74b8462fe0cc9828dfa5ab707ee02c732ff2c8b8))

- Update
  ([`22ed982`](https://github.com/TensorSpeech/TensorFlowASR/commit/22ed982d3a2a7eee8b1020124a7dd51134e36178))

- Update
  ([`b82e0af`](https://github.com/TensorSpeech/TensorFlowASR/commit/b82e0afb8b2611c9f39498d67b4bf4a4054bb7b3))

- Update
  ([`d6fb3b7`](https://github.com/TensorSpeech/TensorFlowASR/commit/d6fb3b731d7c4837e669b1f9db79ae2abeb1fe9e))

- Update
  ([`fdfa5ee`](https://github.com/TensorSpeech/TensorFlowASR/commit/fdfa5ee0c0fd92e11e4477c05b63e2ffe5b73110))

- Update
  ([`987ec7c`](https://github.com/TensorSpeech/TensorFlowASR/commit/987ec7c217f54285441f01a9c947263852864fb0))

- Update
  ([`cd40f14`](https://github.com/TensorSpeech/TensorFlowASR/commit/cd40f147eaf9a52935deef6a5eafc2912d412a22))

- Update
  ([`3efd0dc`](https://github.com/TensorSpeech/TensorFlowASR/commit/3efd0dce225224064b8897a1f3ec60dd84a562f3))

- Update
  ([`f6c2f31`](https://github.com/TensorSpeech/TensorFlowASR/commit/f6c2f3148fd3ac8e2342e391a36d43a26e1f07ab))

- Update
  ([`4e47002`](https://github.com/TensorSpeech/TensorFlowASR/commit/4e470025e312a9cda24f80e24368bed460795cf8))

- Update
  ([`9591496`](https://github.com/TensorSpeech/TensorFlowASR/commit/95914961d0f75312b1c206c9baf8be625374d2a9))

- Update
  ([`18c9458`](https://github.com/TensorSpeech/TensorFlowASR/commit/18c9458b6778e40d3dc36ae570d70d4c37471e71))

- Update
  ([`7dd83cd`](https://github.com/TensorSpeech/TensorFlowASR/commit/7dd83cdc13fb824bd44cedbd8edac5b2c7bfc837))

- Update
  ([`fcafc51`](https://github.com/TensorSpeech/TensorFlowASR/commit/fcafc51bfadfacd3e2f3ff997322a9f240bd6fcc))

- Update
  ([`484fd7c`](https://github.com/TensorSpeech/TensorFlowASR/commit/484fd7cd7f918991eac5b6d639213eca3e82b900))

- Update
  ([`9eef866`](https://github.com/TensorSpeech/TensorFlowASR/commit/9eef8667b2e5c25b3c6654a5ea3fbb0a4dda8569))

- Update
  ([`b7a4bfb`](https://github.com/TensorSpeech/TensorFlowASR/commit/b7a4bfb8982a36ac3e32b0600a80b2d5d537ee1c))

- Update
  ([`d11a967`](https://github.com/TensorSpeech/TensorFlowASR/commit/d11a9679387c28fcc60765be387f97608abb4324))

- Update
  ([`13eb2fd`](https://github.com/TensorSpeech/TensorFlowASR/commit/13eb2fd7bf249ee32f812a3757abf26280ce2824))

- Update
  ([`c6ce439`](https://github.com/TensorSpeech/TensorFlowASR/commit/c6ce43954f937f5a74d3c9d06809e4871b2227ba))

- Update
  ([`550517a`](https://github.com/TensorSpeech/TensorFlowASR/commit/550517af6a5b515239c297014bc05001ec8c7831))

- Update
  ([`0c9a451`](https://github.com/TensorSpeech/TensorFlowASR/commit/0c9a45138624c089a79873df718016d12e2b9792))

- Update
  ([`c6844ce`](https://github.com/TensorSpeech/TensorFlowASR/commit/c6844ce230ab3c7cb526f1e448e7d2e6bbbfea6a))

- Update
  ([`c2448b5`](https://github.com/TensorSpeech/TensorFlowASR/commit/c2448b5d7384457b4cdbb4e1d2b24c702ec554d8))

- Update
  ([`fbf9279`](https://github.com/TensorSpeech/TensorFlowASR/commit/fbf9279196436e252516e616df43d5254882c1d6))

- Update
  ([`81a323f`](https://github.com/TensorSpeech/TensorFlowASR/commit/81a323f8d81bfab3cbc0216a97e9152b00864f5f))

- Update
  ([`a663bef`](https://github.com/TensorSpeech/TensorFlowASR/commit/a663bef7c46f1d72b49d7561184e0d62a877822f))

- Update
  ([`9af0017`](https://github.com/TensorSpeech/TensorFlowASR/commit/9af00171b583a85ae80817830cb170a6a291375b))

- Update
  ([`bb03d39`](https://github.com/TensorSpeech/TensorFlowASR/commit/bb03d39a297b1a78d31d92c5c3ee81da4a89efdc))

- Update
  ([`7c1dbc6`](https://github.com/TensorSpeech/TensorFlowASR/commit/7c1dbc639bf7f449d4e18bae7d431f89791fb11c))

- Update
  ([`9ef7825`](https://github.com/TensorSpeech/TensorFlowASR/commit/9ef7825df73750736d1f139d2baa7aa4e718814f))

- Update
  ([`f560da2`](https://github.com/TensorSpeech/TensorFlowASR/commit/f560da27b4f075ec7b6fa4cd079329685f23f1c9))

- Update
  ([`91c4d48`](https://github.com/TensorSpeech/TensorFlowASR/commit/91c4d48219065a60a0d35c132eb4cdc2274e8062))

- Update
  ([`a1fb85a`](https://github.com/TensorSpeech/TensorFlowASR/commit/a1fb85a2a07616114a60030a79f6befa47ad39ff))

- Update
  ([`8a547e2`](https://github.com/TensorSpeech/TensorFlowASR/commit/8a547e2e4d225ed2759abe697f89dd0ff11b9a56))

- Update
  ([`1ac0bf5`](https://github.com/TensorSpeech/TensorFlowASR/commit/1ac0bf50cb536fc82dfa5b415a593d0381a37573))

- Update
  ([`82a3307`](https://github.com/TensorSpeech/TensorFlowASR/commit/82a330771519c12b9d483b4c49a3ad71d777a79b))

- Update
  ([`f515f40`](https://github.com/TensorSpeech/TensorFlowASR/commit/f515f4014a28cc496c025d637f92772f65047300))

- Update
  ([`0dbffd0`](https://github.com/TensorSpeech/TensorFlowASR/commit/0dbffd02a06dc41b6a1abd93106ee3765346fa07))

- Update
  ([`f9d739c`](https://github.com/TensorSpeech/TensorFlowASR/commit/f9d739c0191e9a2ca74feedbc58ca4ccefbeb34b))

- Update
  ([`b61a993`](https://github.com/TensorSpeech/TensorFlowASR/commit/b61a9935485830b654ceac90b6ce33d27671055a))

- Update
  ([`71db9d5`](https://github.com/TensorSpeech/TensorFlowASR/commit/71db9d5ca6bc8e2ab95801435c7d3edd54aed5c3))

- Update
  ([`e69adbb`](https://github.com/TensorSpeech/TensorFlowASR/commit/e69adbbb0042f46422543c97fc9fb089fe0b02cd))

- Update
  ([`c19ef92`](https://github.com/TensorSpeech/TensorFlowASR/commit/c19ef923ccee9adc0244855b788a6a62c9856287))

- Update
  ([`60a2e8d`](https://github.com/TensorSpeech/TensorFlowASR/commit/60a2e8d947953ef52225d3cf20cddd9a5f449d57))

- Update
  ([`4d95af9`](https://github.com/TensorSpeech/TensorFlowASR/commit/4d95af9a620ee0eacca983b70de33fe24cc174f0))

- Update
  ([`87845a5`](https://github.com/TensorSpeech/TensorFlowASR/commit/87845a57c333ea6f253fe6da695c7ddd0f06740a))

- Update
  ([`7297b8e`](https://github.com/TensorSpeech/TensorFlowASR/commit/7297b8e02252362b6c6ea3a2fcf8b4f760dc24ec))

- Update
  ([`c2d5fef`](https://github.com/TensorSpeech/TensorFlowASR/commit/c2d5fef7f97b9880a4efbae4e5bf7b50fec44e3f))

- Update
  ([`e355485`](https://github.com/TensorSpeech/TensorFlowASR/commit/e3554851bae5ef45ff52f9f8bc1c67d4649f5330))

- Update
  ([`b68eea6`](https://github.com/TensorSpeech/TensorFlowASR/commit/b68eea64acadf0a7192b008bb08a2fe519edd9cf))

- Update
  ([`5377f03`](https://github.com/TensorSpeech/TensorFlowASR/commit/5377f03f99113e9038415c2d32b70a304b6a3d13))

- Update
  ([`cc8692e`](https://github.com/TensorSpeech/TensorFlowASR/commit/cc8692e37dc7d75079133733d9ef0c315ef4a424))

- Update add ga
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update all global shapes to dataset
  ([`b4c1385`](https://github.com/TensorSpeech/TensorFlowASR/commit/b4c138562b581a832d60b9539e1b904d3268832b))

- Update apply decoder gauss weight noise
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update apply gwn
  ([`5641040`](https://github.com/TensorSpeech/TensorFlowASR/commit/5641040f66ec5ee7f69bb14f6ad2123d3062bc27))

- Update apply gwn
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update apply mask
  ([`036cf61`](https://github.com/TensorSpeech/TensorFlowASR/commit/036cf613bea3eb2425ec6d98b0cd895d0adde673))

- Update apply mask
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update asr dataset
  ([`c0b2f66`](https://github.com/TensorSpeech/TensorFlowASR/commit/c0b2f66ec599ea95216501704b7c39424711524b))

- Update asr dataset
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update asr tfrecords dataset
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update attention mask
  ([`c6f2779`](https://github.com/TensorSpeech/TensorFlowASR/commit/c6f2779b569df9b1653df8305d3511006905e509))

- Update attention mask
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update attention mask and rnnt loss logits length
  ([`764a542`](https://github.com/TensorSpeech/TensorFlowASR/commit/764a54272cf09b3ba2bb224c51668a3973739564))

- Update attention mask conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update augment
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update base model train
  ([`35e0958`](https://github.com/TensorSpeech/TensorFlowASR/commit/35e095819642dad2b128936bd3701ecbf2a4ec43))

- Update bfloat16 to float16
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update caching
  ([`2b86069`](https://github.com/TensorSpeech/TensorFlowASR/commit/2b86069513f18e4683dc67afc5ba64056cd9fabf))

- Update callbacks
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Update callbacks and lr schedule
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Update compute output shape
  ([`606f243`](https://github.com/TensorSpeech/TensorFlowASR/commit/606f24372aa96821477e1bedb1e1b0a3169dec6b))

- Update compute output shape
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update compute output shape for dw convs
  ([`7d0a513`](https://github.com/TensorSpeech/TensorFlowASR/commit/7d0a5139875f00eb0273531ac67c940f0193c855))

- Update conf
  ([`212bd3d`](https://github.com/TensorSpeech/TensorFlowASR/commit/212bd3d38ac2feccc903a6d91cd7a17577b0c0d2))

- Update conf
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Update conf
  ([`f8a2cba`](https://github.com/TensorSpeech/TensorFlowASR/commit/f8a2cbafa2ae6f945eeaa0ab4c7c50ac647095c7))

- Update conf
  ([`f5a63bd`](https://github.com/TensorSpeech/TensorFlowASR/commit/f5a63bd574da987a5185d0a42722408bea1aa1a3))

- Update conf
  ([`88be396`](https://github.com/TensorSpeech/TensorFlowASR/commit/88be396308b6ac1f0f6791e69f88f2b1c5159e53))

- Update conf
  ([`6ff244a`](https://github.com/TensorSpeech/TensorFlowASR/commit/6ff244a870159994dae769403de1da94fb7ffae9))

- Update conf
  ([`87fde00`](https://github.com/TensorSpeech/TensorFlowASR/commit/87fde000e6db0cdab6f12a51538b68587cf58f15))

- Update config
  ([`2570e34`](https://github.com/TensorSpeech/TensorFlowASR/commit/2570e34f9b0050a6b2402acf10eb9dd04de24f86))

- Update config
  ([`09d9029`](https://github.com/TensorSpeech/TensorFlowASR/commit/09d9029c9d2b95644c27ad40c92fac8b244242b5))

- Update config
  ([`2d4a221`](https://github.com/TensorSpeech/TensorFlowASR/commit/2d4a2218fb2a8f789cf68c61b1e6fbc8d8c3f9f6))

- Update config
  ([`d0eddb3`](https://github.com/TensorSpeech/TensorFlowASR/commit/d0eddb3771d65da5e2af3cdb86afc4751b9e734e))

- Update config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update config examples
  ([`b3e395b`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3e395b7dab66b0dd31bb38c5cd2b8f9f93a4d10))

- Update config libri
  ([`8efef23`](https://github.com/TensorSpeech/TensorFlowASR/commit/8efef23eb6df7565a15dd51d112b3f62fb8aa533))

- Update conformer + contextnet
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update conformer conf
  ([`56108bc`](https://github.com/TensorSpeech/TensorFlowASR/commit/56108bc258aa5344e33258e2b3daf4b62223614f))

- Update conformer config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update conformer conv module
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update conformer ctc
  ([`20f2f0e`](https://github.com/TensorSpeech/TensorFlowASR/commit/20f2f0e942448a8bb06ca44df4067ba16a4d38d0))

- Update conformer ctc
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update conformer encoder
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update conformer example config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update conformer transducer result
  ([`ae52e7a`](https://github.com/TensorSpeech/TensorFlowASR/commit/ae52e7a00214cb284dcff84f429cc23af2c1abab))

- Update conformer transducer result
  ([`c59a18d`](https://github.com/TensorSpeech/TensorFlowASR/commit/c59a18d60283eb8ca7688833a69990b053679404))

- Update contextnet
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update contextnet conf
  ([`65d4924`](https://github.com/TensorSpeech/TensorFlowASR/commit/65d49246c46a85a95e3deab84f993d25c6923edd))

- Update contextnet conf
  ([`7cdc82e`](https://github.com/TensorSpeech/TensorFlowASR/commit/7cdc82ec7eccfd6faa6a62cf0e0f27d01e6810ac))

- Update contextnet masking
  ([`2284f29`](https://github.com/TensorSpeech/TensorFlowASR/commit/2284f2990c67f9da6454d09ab223819dab9219c2))

- Update conv module for conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update create tfrecords script
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update ctc conformer conf
  ([`a40baf2`](https://github.com/TensorSpeech/TensorFlowASR/commit/a40baf2456226158851124739e7e5c6f86492781))

- Update ctc loss
  ([`9d65bea`](https://github.com/TensorSpeech/TensorFlowASR/commit/9d65bea31925d7fbae5dd42c1c93a92d13a62824))

- Update ctc models
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update dataset
  ([`9803619`](https://github.com/TensorSpeech/TensorFlowASR/commit/9803619e9582e4841c298cdb4718b3a1eea22b80))

- Update dataset
  ([`21c852e`](https://github.com/TensorSpeech/TensorFlowASR/commit/21c852e09d5100b86cf4b8b223c325dc4c5ea499))

- Update decoder gauss noise
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update default config
  ([`da6f4b1`](https://github.com/TensorSpeech/TensorFlowASR/commit/da6f4b1cb320c41b48f95daaa555759dad8cf0b5))

- Update default gen vocab wordpiece
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update default speech config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update default subsampling and attention mask
  ([`84de127`](https://github.com/TensorSpeech/TensorFlowASR/commit/84de127ca0aac680c9d819799669b03139ef657e))

- Update deps
  ([`afc9972`](https://github.com/TensorSpeech/TensorFlowASR/commit/afc9972781072f7d4e333fa3ad12dafef3a8f840))

- Update docstring for relative sinusoidal position encoding
  ([`f3a306b`](https://github.com/TensorSpeech/TensorFlowASR/commit/f3a306b6643e2943515a4ecbfbf69d67fc23f48a))

- Update ds2
  ([`6a187e1`](https://github.com/TensorSpeech/TensorFlowASR/commit/6a187e19b0d534db3d542ae599bb48dab26a1400))

- Update ds2
  ([`363d8c3`](https://github.com/TensorSpeech/TensorFlowASR/commit/363d8c321af550eb912c66e924b4e0999c374fae))

- Update ds2
  ([`f33d257`](https://github.com/TensorSpeech/TensorFlowASR/commit/f33d2573b90986161005decd8f9a6971c703133f))

- Update ds2
  ([`3647de8`](https://github.com/TensorSpeech/TensorFlowASR/commit/3647de844d539f16903e6b7f04e0f81b2fc95bf8))

- Update ds2
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update ds2 conf
  ([`07eb8d1`](https://github.com/TensorSpeech/TensorFlowASR/commit/07eb8d11f1b8d1750e2254cbae72afd2ffbdcd3c))

- Update ds2 dtype
  ([`cd283c0`](https://github.com/TensorSpeech/TensorFlowASR/commit/cd283c0becfb8b30672857472ae4dd49e838fcef))

- Update dtype
  ([`4919627`](https://github.com/TensorSpeech/TensorFlowASR/commit/49196271b8057364da539f66501545c0549a11f2))

- Update dtype
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update dtype for bfloat16 mha
  ([`7daf66a`](https://github.com/TensorSpeech/TensorFlowASR/commit/7daf66a6a78b5b9bdd96af1a2d15de00e0a6c005))

- Update eager
  ([`da5ced3`](https://github.com/TensorSpeech/TensorFlowASR/commit/da5ced358e719e7b0a4456652019e48faad89d94))

- Update env
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update env mxp and cast float32 for rnn if using bfloat16
  ([`90298fb`](https://github.com/TensorSpeech/TensorFlowASR/commit/90298fb1dd372027224ad9b1a390c98bd60e15e7))

- Update env setup
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update env setup device
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update env util setup mxp
  ([`1d2c45a`](https://github.com/TensorSpeech/TensorFlowASR/commit/1d2c45ac5717b913f085a2bb9ff9144be22f2f6c))

- Update eval metadata
  ([`eced9c5`](https://github.com/TensorSpeech/TensorFlowASR/commit/eced9c5b482123e9f3c861f8740ecb8d36bc17c7))

- Update example config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update examples and add steps to training log
  ([`4fec1c5`](https://github.com/TensorSpeech/TensorFlowASR/commit/4fec1c5e7a508421e15c06f8891d90dccc83b58d))

- Update feature extraction layer
  ([`bdb2245`](https://github.com/TensorSpeech/TensorFlowASR/commit/bdb2245bfaed98edc8847e3ccbb4d3c93e9e5537))

- Update featurizer
  ([`d20a796`](https://github.com/TensorSpeech/TensorFlowASR/commit/d20a79651e5c8f84abd8fc857d3f7314ef061704))

- Update file util
  ([`f4ad5b0`](https://github.com/TensorSpeech/TensorFlowASR/commit/f4ad5b0ad8b01320c9cdae4ec8dce13678b66f1e))

- Update function name
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update ga
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Update ga
  ([`af4e5ac`](https://github.com/TensorSpeech/TensorFlowASR/commit/af4e5ac1ccbe6fa905418fe5a63dd9e7962ca890))

- Update ga
  ([`22fd27e`](https://github.com/TensorSpeech/TensorFlowASR/commit/22fd27e0fd766f5f1d78c89a6771822a53d2543b))

- Update ga
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update ga + add wer/cer to validation metrics
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update gauss weight noise
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update gauss weight noise decoder
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update gen vocab wordpiece
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update glu, conv module, layers
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update gradient accumulation optimizer iteration update
  ([`e54a57f`](https://github.com/TensorSpeech/TensorFlowASR/commit/e54a57f902a40778492e24f664db9a8329994a7d))

- Update greedy recognize single
  ([`1928b12`](https://github.com/TensorSpeech/TensorFlowASR/commit/1928b124d962403ed0396bfa37dcd88d549f86fe))

- Update identity layer
  ([`297aa38`](https://github.com/TensorSpeech/TensorFlowASR/commit/297aa38e5ea4bba748b5ca2746cb77ce235968e0))

- Update install external packages scripts
  ([`c2ddc61`](https://github.com/TensorSpeech/TensorFlowASR/commit/c2ddc61e13158526dd73f72239675e7ec77d68c1))

- Update install rnnt loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update install rnnt_loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update jasper
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update keras callback
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update lamdas
  ([`5adb9d2`](https://github.com/TensorSpeech/TensorFlowASR/commit/5adb9d2320e98e35059645ce72ea5bfc86a4d2f7))

- Update layer norms dtype
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update layers
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update librispeech data config
  ([`f62103a`](https://github.com/TensorSpeech/TensorFlowASR/commit/f62103a3ce22846ebf91c972dc60943ca2061415))

- Update load config file
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update load weights in test files
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update logging
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update loss
  ([`a4c217c`](https://github.com/TensorSpeech/TensorFlowASR/commit/a4c217c23aeabfbb097d0b266a172e8dd34f3c53))

- Update loss
  ([`dfbe4bb`](https://github.com/TensorSpeech/TensorFlowASR/commit/dfbe4bb96c2de5770efd22191727d13501b9e65e))

- Update loss
  ([`4233d98`](https://github.com/TensorSpeech/TensorFlowASR/commit/4233d986721ee4463de9b46e406fdac1c2cb2297))

- Update loss
  ([`2d6fae5`](https://github.com/TensorSpeech/TensorFlowASR/commit/2d6fae54e2b1183fb114e054aec9d144de4c60f1))

- Update loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update loss + loss metrics
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update loss aggregation
  ([`2f51256`](https://github.com/TensorSpeech/TensorFlowASR/commit/2f512568d883b8fa87a2ce87b2b06f395b95ec72))

- Update loss computation
  ([`2a9c447`](https://github.com/TensorSpeech/TensorFlowASR/commit/2a9c447d2ffe41c6f7d1269c05027444f5ce5d13))

- Update losses
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update mask computation and overwrite
  ([`86e5820`](https://github.com/TensorSpeech/TensorFlowASR/commit/86e5820ac973767d5ce704e89b7ba4a808123034))

- Update mask fill
  ([`4732dd8`](https://github.com/TensorSpeech/TensorFlowASR/commit/4732dd8f761ac4e3b4696275c752c6f78c81919c))

- Update masked softmax
  ([`2064ad8`](https://github.com/TensorSpeech/TensorFlowASR/commit/2064ad8ed8887b8ed55de9e7accf4095397a8b5b))

- Update masking
  ([`27dbe21`](https://github.com/TensorSpeech/TensorFlowASR/commit/27dbe213442b43626107c5f84f7f578fc2d8dec4))

- Update masking mha
  ([`db8493f`](https://github.com/TensorSpeech/TensorFlowASR/commit/db8493f03ab1fbaa2c993af682e5329c32bc322d))

- Update memory conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update metadata
  ([`fd42400`](https://github.com/TensorSpeech/TensorFlowASR/commit/fd42400316de7d0cc327a07eabf159adda482da1))

- Update metadata default
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update metadata for eval
  ([`610ce78`](https://github.com/TensorSpeech/TensorFlowASR/commit/610ce78e2b105964cb1cd45a61e50cf7c2f7cb94))

- Update metadata of wordpiece librispeech
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update mha
  ([`4c9aba8`](https://github.com/TensorSpeech/TensorFlowASR/commit/4c9aba8434a08f67d78a650f906f6185c6d0d1d2))

- Update mha
  ([`18ad0f2`](https://github.com/TensorSpeech/TensorFlowASR/commit/18ad0f2efcfbd3810d12830f7b90d8370dd0ac41))

- Update mha
  ([`e3344fb`](https://github.com/TensorSpeech/TensorFlowASR/commit/e3344fb5a73e0a33bd85847afd77ccfd11fc9241))

- Update mha
  ([`05c47db`](https://github.com/TensorSpeech/TensorFlowASR/commit/05c47dbb30597a5ba5e105ce1496c3ae3d26ddeb))

- Update mha
  ([`94934fe`](https://github.com/TensorSpeech/TensorFlowASR/commit/94934fe014478eb7ffd3c86bce6e257110dc6f01))

- Update mha dtype
  ([`3a943f9`](https://github.com/TensorSpeech/TensorFlowASR/commit/3a943f9b6e746d6003419ec50d7a72df706d056a))

- Update mha mask
  ([`8bf0983`](https://github.com/TensorSpeech/TensorFlowASR/commit/8bf09835caaaef5f3f404115c7316b5df6cd8f94))

- Update mha mask
  ([`82505a0`](https://github.com/TensorSpeech/TensorFlowASR/commit/82505a019fc98eab1a2ee480563afd6cd97d639a))

- Update mha masking
  ([`738e454`](https://github.com/TensorSpeech/TensorFlowASR/commit/738e454630d4d57249335d488b83159c6f50a50f))

- Update mhra
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update mhsa
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update multihead attention layer
  ([`4111f50`](https://github.com/TensorSpeech/TensorFlowASR/commit/4111f50e16cedfbc87a4cc6e4f889c6d8a0a1d96))

- Update multihead relative attention
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update multiheaded attention
  ([`94127a8`](https://github.com/TensorSpeech/TensorFlowASR/commit/94127a84eb79d5272245c6ab85468fac1912fa69))

- Update multiheaded attention
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update multiheaded attention mask
  ([`60ead9f`](https://github.com/TensorSpeech/TensorFlowASR/commit/60ead9f7fa4b7439dc981064bfc92fe25ac8f495))

- Update mxp and requirements
  ([`8f9683f`](https://github.com/TensorSpeech/TensorFlowASR/commit/8f9683f751cd9e21ebe0a549c0f558a7cdfcd428))

- Update naive rnnt loss
  ([`40c341e`](https://github.com/TensorSpeech/TensorFlowASR/commit/40c341edb1190b53c879e27791d6a43861046bce))

- Update naive rnnt loss and grad
  ([`d21c5a2`](https://github.com/TensorSpeech/TensorFlowASR/commit/d21c5a2026e56f524e68d717ad60983d8a2e3584))

- Update name layer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update normalization
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update num class
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update optimizer
  ([`f407d5a`](https://github.com/TensorSpeech/TensorFlowASR/commit/f407d5ae4f27551a649a6910ae6cb3c7bfc9ccce))

- Update order of dataset cache
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update output shape
  ([`a9fd973`](https://github.com/TensorSpeech/TensorFlowASR/commit/a9fd973d4f8ad47acb31cfc2faad2439bdccaf35))

- Update output shape
  ([`cbf0791`](https://github.com/TensorSpeech/TensorFlowASR/commit/cbf07919cf480f2660fcb55ed6a09b3d37626ff5))

- Update pad tfarray
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update pe
  ([`ad4e068`](https://github.com/TensorSpeech/TensorFlowASR/commit/ad4e068c106cc45ae3df4d2635eb1d753a52f7c8))

- Update pe
  ([`317ba3b`](https://github.com/TensorSpeech/TensorFlowASR/commit/317ba3b98a7df0682ed40a461f2de8aec424dffb))

- Update pe
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update pe and mha
  ([`db0d2fc`](https://github.com/TensorSpeech/TensorFlowASR/commit/db0d2fcb3a835b7f595af4ad1d2a928bdfbfd49f))

- Update positional encoding and relative shift
  ([`266326e`](https://github.com/TensorSpeech/TensorFlowASR/commit/266326ee9ea875bca5cf47e80326cb3890c8d8ad))

- Update predict logger callback
  ([`be98d4a`](https://github.com/TensorSpeech/TensorFlowASR/commit/be98d4a854b71ac87bfaeca1c4f0206f8bf408a4))

- Update prediction transducer layer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update prepare scripts
  ([`636bbab`](https://github.com/TensorSpeech/TensorFlowASR/commit/636bbab87b4ea01aa01ecb580dc3879afae79bd7))

- Update pretrain model link
  ([`bda2874`](https://github.com/TensorSpeech/TensorFlowASR/commit/bda2874c4da89677138cb6e792d85f8241508b12))

- Update read entries of asr dataset
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update readme
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update reduce replicas losses
  ([`fcfa8b0`](https://github.com/TensorSpeech/TensorFlowASR/commit/fcfa8b0872db5e4b726f0ee0032aeae471ab0090))

- Update regularizer pe
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update rel left shift
  ([`a64fd99`](https://github.com/TensorSpeech/TensorFlowASR/commit/a64fd998302b27719550fa12caf6b9fa16f64af4))

- Update rel shift
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update relative positional encoding and relative shift
  ([`44c49f1`](https://github.com/TensorSpeech/TensorFlowASR/commit/44c49f13af00c61165e15c965335cdab3af5924d))

- Update relative positional encoding, relative multihead attention
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update relmha
  ([`452a854`](https://github.com/TensorSpeech/TensorFlowASR/commit/452a854b317ac10548ab83ccd0904647e62bbc67))

- Update relpe
  ([`83d6680`](https://github.com/TensorSpeech/TensorFlowASR/commit/83d6680b48a7942d6691b828ae709331aa2c4787))

- Update relpe
  ([`b6ca344`](https://github.com/TensorSpeech/TensorFlowASR/commit/b6ca34495ad6826fe6d6faaa22f7fb858abdf9da))

- Update relpe and conformer
  ([`cbacd6f`](https://github.com/TensorSpeech/TensorFlowASR/commit/cbacd6fe10e1f67e171f27bac180ada427410070))

- Update reqs
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update reqs + tflite
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update requirements
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update reset memory states
  ([`a1b881f`](https://github.com/TensorSpeech/TensorFlowASR/commit/a1b881f0c9b036a6f2c31fd02a24136e260d8366))

- Update results contextnet
  ([`564784d`](https://github.com/TensorSpeech/TensorFlowASR/commit/564784d31fe31969143dd1f7e514ce973f803aef))

- Update rnn layers for bfloat16
  ([`da3ddfc`](https://github.com/TensorSpeech/TensorFlowASR/commit/da3ddfc236af1c61fbeb968b7fd7ac078cafe7ed))

- Update rnn transducer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update rnnt
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update rnnt conf
  ([`b720aa4`](https://github.com/TensorSpeech/TensorFlowASR/commit/b720aa4f4b014b518721ac52e659a631bb7ad4fd))

- Update rnnt config
  ([`bae5fa5`](https://github.com/TensorSpeech/TensorFlowASR/commit/bae5fa5e72774d78c230cd11ddd3a9ba5cb7d517))

- Update rnnt loss
  ([`dd0d365`](https://github.com/TensorSpeech/TensorFlowASR/commit/dd0d3657ba427f3a40d97ab0713bb498e70006a1))

- Update rnnt loss
  ([`2f62d52`](https://github.com/TensorSpeech/TensorFlowASR/commit/2f62d526d0b5642d2c4f05ce526f298fc62af863))

- Update rnnt loss
  ([`8404447`](https://github.com/TensorSpeech/TensorFlowASR/commit/84044479e9fae91bc61975eb66a375c04174c562))

- Update rnnt loss
  ([`f07e9ce`](https://github.com/TensorSpeech/TensorFlowASR/commit/f07e9ce9059a6e54736170e0b4afdb6c0b150f5e))

- Update rnnt loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update rnnt loss logits length
  ([`129f8e0`](https://github.com/TensorSpeech/TensorFlowASR/commit/129f8e0bf94291c3610c2d071c146868d708ef4f))

- Update rnnt loss naive
  ([`4008678`](https://github.com/TensorSpeech/TensorFlowASR/commit/4008678c8bbbe8d8d739453e3839825a9f9479d8))

- Update rnnt test
  ([`d3e4773`](https://github.com/TensorSpeech/TensorFlowASR/commit/d3e47736bec1c36e67ac7f2913bf287112d23a20))

- Update schedules
  ([`099d7b3`](https://github.com/TensorSpeech/TensorFlowASR/commit/099d7b3a36a6dd21106d260421fc99da6457edf3))

- Update schedules
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update script gen libri transcripts
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update script train conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update scripts and char vocab build
  ([`b8d327b`](https://github.com/TensorSpeech/TensorFlowASR/commit/b8d327ba85bd34b1395940eb4afa87ccb2a6c3a1))

- Update sentencepiece metadata
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update sentencepiece tft lib
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update setup env
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update setup mxp
  ([`11f2f50`](https://github.com/TensorSpeech/TensorFlowASR/commit/11f2f50f48b00b83de16e0faca60b82d618054c4))

- Update shuffle buffer size
  ([`708908f`](https://github.com/TensorSpeech/TensorFlowASR/commit/708908fc804969570df50c31ec52aa70fdaa98c8))

- Update slice
  ([`2725381`](https://github.com/TensorSpeech/TensorFlowASR/commit/2725381bcfd087e97fd71175a2aa1c2e49fc1577))

- Update softmax masking
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update softmax mha
  ([`3a1dcda`](https://github.com/TensorSpeech/TensorFlowASR/commit/3a1dcda3f502f3f632f9be7efc0ec52e99243c66))

- Update sp
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Update sp options
  ([`f2a192d`](https://github.com/TensorSpeech/TensorFlowASR/commit/f2a192d8c3b5247ac0ff273cc6675c1931c6d732))

- Update specaugment
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Update speech config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update speech featurizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update step loss
  ([`f57c2f3`](https://github.com/TensorSpeech/TensorFlowASR/commit/f57c2f3de233b408d93df8e92412602f3c9c392d))

- Update subsampling in conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update subsampling layer
  ([`f17913a`](https://github.com/TensorSpeech/TensorFlowASR/commit/f17913a52aeb989ceb71700505468cc88b21b335))

- Update summary of models to use the expand nested one
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update test cases and docs
  ([`95f6ba5`](https://github.com/TensorSpeech/TensorFlowASR/commit/95f6ba5c4860ceab846534beef05145299dba377))

- Update test save model and random
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update tf logging
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update tfrecords dataset create
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update time reduction and restructure examples
  ([`cfc9a77`](https://github.com/TensorSpeech/TensorFlowASR/commit/cfc9a775d07a67dc39839ca87148b97d95bba0ac))

- Update train log steps
  ([`9867ebb`](https://github.com/TensorSpeech/TensorFlowASR/commit/9867ebb143877c161d22c897397b037f9a0487aa))

- Update train scripts
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update train/test script with mxp
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update transducer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update transducer models
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update transducer models with positional encoding
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update transducer prediction
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update transducer recognize
  ([`6b8fe03`](https://github.com/TensorSpeech/TensorFlowASR/commit/6b8fe034cb0b38c29f2385196038968d0b20e14b))

- Update vgg subsampling
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update vocab size
  ([`26ed783`](https://github.com/TensorSpeech/TensorFlowASR/commit/26ed7834e904956786e763ea738bf1ea8004e070))

- Update vocab size setter
  ([`410ff46`](https://github.com/TensorSpeech/TensorFlowASR/commit/410ff4655ada20ff0bbaf236ec0568c8266b4d57))

- Update wordpiece
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update wordpiece featurizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update wordpiece featurizer extract
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update wordpiece gen + librispeech wordpiece
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update wordpiece metadata
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update wordpiece tokenizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update wp metadata
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Updatee dtype
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Use concat instead ragged tensor
  ([`495a4e5`](https://github.com/TensorSpeech/TensorFlowASR/commit/495a4e57d6420dacce9d2c24b7c96652ff71f978))

- Use cpu for rnnt loss
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Use default recurrent layer
  ([`4295974`](https://github.com/TensorSpeech/TensorFlowASR/commit/42959747efb85a62eaa0b1a1efb9a5e8136bdac7))

- Use env bash
  ([`239245f`](https://github.com/TensorSpeech/TensorFlowASR/commit/239245fe3ee0f0e45c2668b6e2a143eb614d313e))

- Use gradient accumulation with gradient noise
  ([`b6710d5`](https://github.com/TensorSpeech/TensorFlowASR/commit/b6710d587cb540f79c4afea2b6e08e7f9c8a8824))

- Use per layer mha attention bias
  ([`c70a2e8`](https://github.com/TensorSpeech/TensorFlowASR/commit/c70a2e891042f3abb86b72fc2bfd27fc32c9f4b6))

- Use ragged tensor for padding
  ([`f3e05fe`](https://github.com/TensorSpeech/TensorFlowASR/commit/f3e05fe3fe582f8b7cfeee698fc71b99b65496bd))

- Use rnn with bfloat16
  ([`bdd1b40`](https://github.com/TensorSpeech/TensorFlowASR/commit/bdd1b40d62069c17315e49d3f051b0851884ddf5))

- Use slice instead of zeros
  ([`0ab6d1d`](https://github.com/TensorSpeech/TensorFlowASR/commit/0ab6d1d6a9fec372a7c9cc186149413b30404e0d))

- Use sparse labels ctc loss
  ([`e24604b`](https://github.com/TensorSpeech/TensorFlowASR/commit/e24604b4a86d007d3efbbc0cddcaf19bf7863e10))

- Use sparse labels ctc loss not in tpu
  ([`99121d5`](https://github.com/TensorSpeech/TensorFlowASR/commit/99121d59a69c8cb88213b63526713556a731bbae))

- Use static shape rnnt loss
  ([`519331f`](https://github.com/TensorSpeech/TensorFlowASR/commit/519331fa316a609d7d696fd68384cd48fa1623e8))

- Use tf range
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Vgg blurpool subsampling
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Watch tape
  ([`32b52b8`](https://github.com/TensorSpeech/TensorFlowASR/commit/32b52b8aade910cbc9c5fd3a8726cfce18cbcafb))

- Watch y_pred
  ([`2233e43`](https://github.com/TensorSpeech/TensorFlowASR/commit/2233e43f981e00b9b1f46c6ae2530ee525c673e3))

- Wordpiece featurizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Wp features classes
  ([`28db7e3`](https://github.com/TensorSpeech/TensorFlowASR/commit/28db7e317314b7734f47243629a9f74369ed17a3))

- Wp featurizer vocab size
  ([`d69c4ac`](https://github.com/TensorSpeech/TensorFlowASR/commit/d69c4ace28964e3cbc2bcd58a7aa24431d55a0d9))

- Wrong loss metric update state
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Wrong mhsa type in conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Wrong type addition
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

### Chores

- Add character metadata
  ([`ad4cef1`](https://github.com/TensorSpeech/TensorFlowASR/commit/ad4cef19725b0ed9e695b995b3ad8f79ea8088db))

- Add some wordpieces
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add support for tf 2.12.0
  ([`567bd95`](https://github.com/TensorSpeech/TensorFlowASR/commit/567bd95edbdfda5150af8bbcef8db06482cc443c))

- Add tf 2.14
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Buffer size
  ([`692199c`](https://github.com/TensorSpeech/TensorFlowASR/commit/692199c7d14c98dabe9b8671ebd1494d7398879c))

- Bump 2.0.0
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Char ctc conformer
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Character libri metadata
  ([`045ee32`](https://github.com/TensorSpeech/TensorFlowASR/commit/045ee32cc01b2a7abd4f5adb886c21cb5dabaed4))

- Conf
  ([`b8794ce`](https://github.com/TensorSpeech/TensorFlowASR/commit/b8794ce7b313bf1322ac6c6cd7dba6c760561be4))

- Conf
  ([`c993110`](https://github.com/TensorSpeech/TensorFlowASR/commit/c99311024ea79325f72b216bb4f56f7b8119feba))

- Conf
  ([`b4ccaf7`](https://github.com/TensorSpeech/TensorFlowASR/commit/b4ccaf7c5eece324838b738f0c3460d80abac4b4))

- Conf
  ([`d6619a5`](https://github.com/TensorSpeech/TensorFlowASR/commit/d6619a51679efe3245a59b11f5312ee56ee6f8ef))

- Conf
  ([`c4b95d7`](https://github.com/TensorSpeech/TensorFlowASR/commit/c4b95d7440f93094f21c365ee81e6be49e217b82))

- Conf
  ([`47a8403`](https://github.com/TensorSpeech/TensorFlowASR/commit/47a84036401af6ef2a1d7081ef9367c1136d5907))

- Conf
  ([`b740623`](https://github.com/TensorSpeech/TensorFlowASR/commit/b740623601d62530efebda169188ea082d5b7799))

- Conf
  ([`3993058`](https://github.com/TensorSpeech/TensorFlowASR/commit/39930588daf9e7bbd6e992ae547d88fdcb229af1))

- Config
  ([`76bfba8`](https://github.com/TensorSpeech/TensorFlowASR/commit/76bfba8870143c15916d7197c99b0c51fce40b2a))

- Config
  ([`74b7ffd`](https://github.com/TensorSpeech/TensorFlowASR/commit/74b7ffd6f47d4d647d8dd4076b7c4951afa6b8ee))

- Configs
  ([`2b44f63`](https://github.com/TensorSpeech/TensorFlowASR/commit/2b44f6344cdd930e4c8216e8bfd50f169b5ce810))

- Configs sp
  ([`6716fc2`](https://github.com/TensorSpeech/TensorFlowASR/commit/6716fc22af63f1febf1e624364f4813790465c09))

- Conformer conf
  ([`e92de97`](https://github.com/TensorSpeech/TensorFlowASR/commit/e92de972fc11b406ed97db0a389055acad80c478))

- Conformer conf
  ([`6a4ce32`](https://github.com/TensorSpeech/TensorFlowASR/commit/6a4ce32f652ea0dea84fc2c946f058690a22b4cf))

- Conformer conf
  ([`1ec5426`](https://github.com/TensorSpeech/TensorFlowASR/commit/1ec5426397389a524bc099c2dfe23c1ca1ccf4a0))

- Conformer ctc conf
  ([`524d045`](https://github.com/TensorSpeech/TensorFlowASR/commit/524d04525e15814cc40eb762d55737020977f72a))

- Conformer specaug conf
  ([`99ceeac`](https://github.com/TensorSpeech/TensorFlowASR/commit/99ceeac2885d37281a877db7c72cb3e00dbad489))

- Contextnet conf
  ([`04b8172`](https://github.com/TensorSpeech/TensorFlowASR/commit/04b8172371785f1741e782b290daa302dbfa7e4d))

- Contextnet conf
  ([`bc7c030`](https://github.com/TensorSpeech/TensorFlowASR/commit/bc7c0301be690fe229eb4ea1843161d2471c9eca))

- Ctc conformer config
  ([`07c2b65`](https://github.com/TensorSpeech/TensorFlowASR/commit/07c2b65d7821d81a5506f24e307cb34a7dc5435a))

- Data cache
  ([`01dfab0`](https://github.com/TensorSpeech/TensorFlowASR/commit/01dfab0e24124642d849ee595ed32ca9e6a2b5a6))

- Debug
  ([`2ce3eca`](https://github.com/TensorSpeech/TensorFlowASR/commit/2ce3eca5eca9fd0dbb9c1a96d49e247c5ff695a2))

- Debug
  ([`a46a0d9`](https://github.com/TensorSpeech/TensorFlowASR/commit/a46a0d9a4a41cff24edc488e710354f4e8f43ee4))

- Debug
  ([`66e07a3`](https://github.com/TensorSpeech/TensorFlowASR/commit/66e07a39a9a4b84e07304bad3f771e17e5306b43))

- Debug
  ([`6ac46f7`](https://github.com/TensorSpeech/TensorFlowASR/commit/6ac46f7268453418dc5a077f71d198e704ba360f))

- Debug
  ([`04ef71c`](https://github.com/TensorSpeech/TensorFlowASR/commit/04ef71ca525c1c37f5ab2cae364f619db86ccc71))

- Debug
  ([`d5c0f1f`](https://github.com/TensorSpeech/TensorFlowASR/commit/d5c0f1f610ba5379a90b1ca4257a9400f8b1f909))

- Deps
  ([`6525552`](https://github.com/TensorSpeech/TensorFlowASR/commit/65255521e4902474b405a989af28632eb16f2302))

- Dtype
  ([`9b77663`](https://github.com/TensorSpeech/TensorFlowASR/commit/9b77663566e7b506fecdf77d9b4549b99139d211))

- Dtype policy
  ([`3fb0d53`](https://github.com/TensorSpeech/TensorFlowASR/commit/3fb0d535d480858b5bcab584dbaa7c30d003e7e0))

- Fix attention mask
  ([`d4e77ea`](https://github.com/TensorSpeech/TensorFlowASR/commit/d4e77eace441863ddae4791edc7c09249561f5cc))

- Fix configs
  ([`3962772`](https://github.com/TensorSpeech/TensorFlowASR/commit/3962772cc6ca4092f383540dcbd35272eeb4b575))

- Fix deps and vscode configs
  ([`9a59241`](https://github.com/TensorSpeech/TensorFlowASR/commit/9a59241b7cfdf161fef1e84972b669df8d5df789))

- Fix test batch size assignment
  ([`5613867`](https://github.com/TensorSpeech/TensorFlowASR/commit/5613867c0aaeae16c303e475647627dd2d80de0a))

- Init docs
  ([`bee65d3`](https://github.com/TensorSpeech/TensorFlowASR/commit/bee65d31a4962514c3f30f5ad7b6f014e6c36ffe))

- Mha masked softmax
  ([`77e69ce`](https://github.com/TensorSpeech/TensorFlowASR/commit/77e69ce91868f65310b8630b39368c92e5b3f11e))

- Mha masked softmax
  ([`e663982`](https://github.com/TensorSpeech/TensorFlowASR/commit/e6639829b8767ae00d3920ee9ba60dd67aa6a354))

- Reformat
  ([`0499b47`](https://github.com/TensorSpeech/TensorFlowASR/commit/0499b47c96241d4c14b4eb8127c635ccab94af99))

- Remove casting rnn
  ([`9f35b3a`](https://github.com/TensorSpeech/TensorFlowASR/commit/9f35b3a12fd8163f0295e80c6623d0dbebdef34f))

- Remove print
  ([`2c5e106`](https://github.com/TensorSpeech/TensorFlowASR/commit/2c5e1064b8d683dbc95c98a6697dfab0b42592b9))

- Revert data cache
  ([`bfa761b`](https://github.com/TensorSpeech/TensorFlowASR/commit/bfa761bdca7a934568c9fd73fbaf83d70b43d213))

- Revert rnn types
  ([`1c06cca`](https://github.com/TensorSpeech/TensorFlowASR/commit/1c06ccade40125db1818aa21d1342d38f4ae4991))

- Rnnt tiny
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Sentencepiece whitespace vocab libri
  ([`e5cdd87`](https://github.com/TensorSpeech/TensorFlowASR/commit/e5cdd87fa4cbdc780f53852bf534028d7f10dbb0))

- Sentencepiece whitespace vocab libri
  ([`d748b8d`](https://github.com/TensorSpeech/TensorFlowASR/commit/d748b8d876f68af5c79b8c2756f873ab4a815d70))

- Small update
  ([`5371ae3`](https://github.com/TensorSpeech/TensorFlowASR/commit/5371ae3e637e00b7d17279c266077a777b04b08f))

- Specaugment for ds2
  ([`c5657da`](https://github.com/TensorSpeech/TensorFlowASR/commit/c5657da21c2b851594cfd1bdaf9b9d908c71e0b1))

- Surpress deprecation warnings
  ([`3d68321`](https://github.com/TensorSpeech/TensorFlowASR/commit/3d6832120fb25e3dd92b6157c951be860b825b40))

- Test
  ([`391e359`](https://github.com/TensorSpeech/TensorFlowASR/commit/391e359bbafa1392b13fe79adc81591ae8bb7348))

- Test
  ([`5a044c4`](https://github.com/TensorSpeech/TensorFlowASR/commit/5a044c49e0afa33afc4e77cb5f2357c05ff073fb))

- Test
  ([`251bfbe`](https://github.com/TensorSpeech/TensorFlowASR/commit/251bfbea18493d5ff9532aca24fa63285344594d))

- Test
  ([`468e485`](https://github.com/TensorSpeech/TensorFlowASR/commit/468e4855b75c95188eb3c1c442ceaad69763289f))

- Test
  ([`9de009a`](https://github.com/TensorSpeech/TensorFlowASR/commit/9de009ad55fac192c50b1475fcd612004a5d6541))

- Test
  ([`fd6b6c2`](https://github.com/TensorSpeech/TensorFlowASR/commit/fd6b6c2adc1f7a43463fd515d28caed2df032b48))

- Test no caching
  ([`14067a2`](https://github.com/TensorSpeech/TensorFlowASR/commit/14067a261a8ae6aee117418020701587ff245ed4))

- Test slicing fix
  ([`cb50172`](https://github.com/TensorSpeech/TensorFlowASR/commit/cb501729aec24b2707ed7245b06e3700bea46436))

- Test speech featurizer in model
  ([`04180a5`](https://github.com/TensorSpeech/TensorFlowASR/commit/04180a5417f0ac99627f96de4c2ee148c8d856f3))

- Tf2.15
  ([`b2731d3`](https://github.com/TensorSpeech/TensorFlowASR/commit/b2731d34c1e6647892d9dd7d93a45cebe17ff772))

- Tiny rnnt
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Update
  ([`32354df`](https://github.com/TensorSpeech/TensorFlowASR/commit/32354dfa6538a48bc4c4de6a390eef7f1545a07a))

- Update
  ([`5a58cae`](https://github.com/TensorSpeech/TensorFlowASR/commit/5a58cae49d9abfbe033c9a46d28daf213837df01))

- Update
  ([`3515d71`](https://github.com/TensorSpeech/TensorFlowASR/commit/3515d7129d30a11b187bc462e716bcdf694c7761))

- Update
  ([`de06fc5`](https://github.com/TensorSpeech/TensorFlowASR/commit/de06fc5c4f5411b333fb66b85c135ec7bf32b1c8))

- Update
  ([`34e9842`](https://github.com/TensorSpeech/TensorFlowASR/commit/34e98421234ac1367ae9d82340cebe8ec6e5cd42))

- Update
  ([`3108fe6`](https://github.com/TensorSpeech/TensorFlowASR/commit/3108fe66f03038b0d4dab82dd7570c02ee914058))

- Update
  ([`ea4b838`](https://github.com/TensorSpeech/TensorFlowASR/commit/ea4b838970355a8b7bc1fb653eefec1f09745430))

- Update
  ([`c6590fe`](https://github.com/TensorSpeech/TensorFlowASR/commit/c6590fe7cd45cb8dca9347f08f0ba251d1ff482d))

- Update
  ([`fe66997`](https://github.com/TensorSpeech/TensorFlowASR/commit/fe66997e94201a2a6b77803b8948ec6addaa15a7))

- Update
  ([`c128f4f`](https://github.com/TensorSpeech/TensorFlowASR/commit/c128f4f797da2d5020cdc4ce08e6bbea224d9088))

- Update
  ([`65f75e9`](https://github.com/TensorSpeech/TensorFlowASR/commit/65f75e9b1bf7ea6210be1f1231b31ad5d399fe3d))

- Update
  ([`47d59de`](https://github.com/TensorSpeech/TensorFlowASR/commit/47d59def4009b236ecd68f23eb665d1469f3afe6))

- Update
  ([`3e26d17`](https://github.com/TensorSpeech/TensorFlowASR/commit/3e26d1711e66a490b055053ca7a0cf1007b03469))

- Update
  ([`18cf15e`](https://github.com/TensorSpeech/TensorFlowASR/commit/18cf15eb5a7c5755ad709f053c2de9befc997528))

- Update
  ([`a7163f9`](https://github.com/TensorSpeech/TensorFlowASR/commit/a7163f997e73bb671a4b2fd7b7e4a0c6841b895c))

- Update
  ([`9fb6ea9`](https://github.com/TensorSpeech/TensorFlowASR/commit/9fb6ea9f731bb5d1f7b1275b7f4bc46f44dac145))

- Update
  ([`9ae01a4`](https://github.com/TensorSpeech/TensorFlowASR/commit/9ae01a480fe6f25baf87c539557ef9d9c5f0b9be))

- Update
  ([`270c88c`](https://github.com/TensorSpeech/TensorFlowASR/commit/270c88cc3f452506216cef3b8c107c9941bb3217))

- Update bn conformer
  ([`da3b8bb`](https://github.com/TensorSpeech/TensorFlowASR/commit/da3b8bb3106717fb6837e451d04cbccca7c3eb53))

- Update conf
  ([`f705997`](https://github.com/TensorSpeech/TensorFlowASR/commit/f705997e981d78f8ed1c0f4b472e2f81d8a67459))

- Update conf
  ([`1b7c77e`](https://github.com/TensorSpeech/TensorFlowASR/commit/1b7c77e3c6efa4e1e5efd91cfb4de1156e45546f))

- Update conf
  ([`6c735e4`](https://github.com/TensorSpeech/TensorFlowASR/commit/6c735e488d70ea0c88da63ac0d627d846d83b022))

- Update conf
  ([`411c70b`](https://github.com/TensorSpeech/TensorFlowASR/commit/411c70bc0b121782fed57f5689713920e85424d6))

- Update conf
  ([`c8a6264`](https://github.com/TensorSpeech/TensorFlowASR/commit/c8a6264ee6aadfdd584636876588757259ecb946))

- Update configs
  ([`68167bd`](https://github.com/TensorSpeech/TensorFlowASR/commit/68167bd83da7f50f9f12fdf3e679f3f42a0c3af5))

- Update confs
  ([`6290c17`](https://github.com/TensorSpeech/TensorFlowASR/commit/6290c179b072d4c37b4f0b6b00fb74f026fc7034))

- Update contextnet results
  ([`8d00cf4`](https://github.com/TensorSpeech/TensorFlowASR/commit/8d00cf4b73f9d00219b8efdbec16a34c33796828))

- Update ctc conformer
  ([`00b97fa`](https://github.com/TensorSpeech/TensorFlowASR/commit/00b97fa5541cdf98d5272178e26b4ab578d82c26))

- Update ctc loss
  ([`b37efe2`](https://github.com/TensorSpeech/TensorFlowASR/commit/b37efe2b193309be679317ff3f8251984437ce40))

- Update documentation
  ([`a19f585`](https://github.com/TensorSpeech/TensorFlowASR/commit/a19f5857f8d8a01e34f6966a488f689d3bf1fee5))

- Update ds2
  ([`746b4ae`](https://github.com/TensorSpeech/TensorFlowASR/commit/746b4ae97358ce783fa822ef5793eb11ab9e6d53))

- Update ds2 conf
  ([`7ba27c2`](https://github.com/TensorSpeech/TensorFlowASR/commit/7ba27c2573d709fd8af8045f9296ef80b975d0b6))

- Update ds2 conf
  ([`f309039`](https://github.com/TensorSpeech/TensorFlowASR/commit/f30903984fcf9ed8d0c579f6bf716eb05c181d9e))

- Update examples
  ([`09730cf`](https://github.com/TensorSpeech/TensorFlowASR/commit/09730cfad174fcbea1311e8c93c960a2efd83321))

- Update file util
  ([`cc973c6`](https://github.com/TensorSpeech/TensorFlowASR/commit/cc973c6bccbf315325946b4384dcf5b771ca7728))

- Update gen vocab and meta
  ([`a45d57a`](https://github.com/TensorSpeech/TensorFlowASR/commit/a45d57a3be677763e341ab51b65f48f8fd30c7c4))

- Update getattr
  ([`a6b027f`](https://github.com/TensorSpeech/TensorFlowASR/commit/a6b027fe893e89322885e560086ef7b6fea3eb2d))

- Update librispeech
  ([`4be75bb`](https://github.com/TensorSpeech/TensorFlowASR/commit/4be75bbf117feb44d283187411ea65879f26961b))

- Update log
  ([`b3af72a`](https://github.com/TensorSpeech/TensorFlowASR/commit/b3af72a331d9d12fb10439c5c2bfd78580c00a17))

- Update log
  ([`a6d9b50`](https://github.com/TensorSpeech/TensorFlowASR/commit/a6d9b50d9b8072101d2d924d03ab759c206b3ae9))

- Update math util
  ([`ef58a04`](https://github.com/TensorSpeech/TensorFlowASR/commit/ef58a040033da2f9d059077358b28c70aea1a1d0))

- Update metadata
  ([`e6a1bd2`](https://github.com/TensorSpeech/TensorFlowASR/commit/e6a1bd2e4314639b014797aaf10b9f8647cd421c))

- Update mha mask
  ([`fbf981b`](https://github.com/TensorSpeech/TensorFlowASR/commit/fbf981b2048df800036c11013051aee8b4fa5235))

- Update model conf
  ([`07078f4`](https://github.com/TensorSpeech/TensorFlowASR/commit/07078f453548299b31fabdad84cc4e03ddd688ec))

- Update model conf
  ([`ee648c9`](https://github.com/TensorSpeech/TensorFlowASR/commit/ee648c9fb4dd3870edd5d45411e377cf3e8dc640))

- Update model conf ds2
  ([`a5717a7`](https://github.com/TensorSpeech/TensorFlowASR/commit/a5717a7863ed134b75c776f74519c8f2b35c9160))

- Update output shape conformer
  ([`3033c28`](https://github.com/TensorSpeech/TensorFlowASR/commit/3033c2823fcd095593c13ab08c0620a602464a2a))

- Update preprocess path with option check_exists
  ([`f06368c`](https://github.com/TensorSpeech/TensorFlowASR/commit/f06368c69702c69ea927d29e58e6127e03790e10))

- Update pylint
  ([`0b70e49`](https://github.com/TensorSpeech/TensorFlowASR/commit/0b70e4973b17db39678f5ddb1df755885bc179c3))

- Update python-publish.yml
  ([`c30d62c`](https://github.com/TensorSpeech/TensorFlowASR/commit/c30d62c8dc6433a9bfb560f4235e005b33f65d38))

- Update reqs
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Update requirements for apple m1
  ([`dcb752c`](https://github.com/TensorSpeech/TensorFlowASR/commit/dcb752cda59218679f95a70c6547c14f88a01218))

- Update results
  ([`abe6683`](https://github.com/TensorSpeech/TensorFlowASR/commit/abe66836b878b229902ee1068f87eafe37ef0bb3))

- Update results contextnet
  ([`04a85ef`](https://github.com/TensorSpeech/TensorFlowASR/commit/04a85efc9b29ff8a2e96d213d2d248f55b22362a))

- Update rnnt conf
  ([`a293eca`](https://github.com/TensorSpeech/TensorFlowASR/commit/a293eca693d0770bf62b9221e9810a2dcf829637))

- Update rnnt conf
  ([`eb2010b`](https://github.com/TensorSpeech/TensorFlowASR/commit/eb2010b28d247d4830f6d37717b7bf1adeba9d98))

- Update rnnt conf
  ([`f2c23ca`](https://github.com/TensorSpeech/TensorFlowASR/commit/f2c23caad572d443e6f9643e9fc2ad066457569e))

- Update rnnt conf
  ([`2195881`](https://github.com/TensorSpeech/TensorFlowASR/commit/2195881fe19a4c4f67fe33e6ca78a528a021d19b))

- Update rnnt conf
  ([`0a9d1fc`](https://github.com/TensorSpeech/TensorFlowASR/commit/0a9d1fca8ede2d074cb48232f2b9955ecb24731b))

- Update rnnt conf
  ([`72c5f8d`](https://github.com/TensorSpeech/TensorFlowASR/commit/72c5f8d2ff2cb19ab6b572463d6bb45e4b9d6584))

- Update rnnt conf
  ([`02d585d`](https://github.com/TensorSpeech/TensorFlowASR/commit/02d585d1a1be33e502cdf55c77a5d7cc3e791646))

- Update rnnt loss
  ([`c41d8de`](https://github.com/TensorSpeech/TensorFlowASR/commit/c41d8de201bb9b3c3e7dd211bf9e0ea3efdf1b88))

- Update rnnt loss
  ([`ba869b5`](https://github.com/TensorSpeech/TensorFlowASR/commit/ba869b53eb5dc7221684039b58a6ebfd875a1e90))

- Update rnnt resutls
  ([`7e15486`](https://github.com/TensorSpeech/TensorFlowASR/commit/7e154868180b8f50f1ec0210394440630f23b50a))

- Update rnnt train
  ([`3ec6394`](https://github.com/TensorSpeech/TensorFlowASR/commit/3ec639407bf7aff57cda8027d471582f836ecfe2))

- Update some example configs
  ([`c4867f6`](https://github.com/TensorSpeech/TensorFlowASR/commit/c4867f62087598edd5b66cfe77e3e12497cc3616))

- Update sp metadata libri
  ([`f66667b`](https://github.com/TensorSpeech/TensorFlowASR/commit/f66667bbd4cfee7f05dd92d1ca089c247927dec2))

- Update sp whitespace model
  ([`abb5703`](https://github.com/TensorSpeech/TensorFlowASR/commit/abb5703d22c5e00dc43c1b9e3c5efa70415608d3))

- Update step outputs
  ([`2326b06`](https://github.com/TensorSpeech/TensorFlowASR/commit/2326b060e26ac93f837d3c867a92ee6c93ed8b25))

- Update test case
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Update vocab
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Vocab librispeech
  ([`526db48`](https://github.com/TensorSpeech/TensorFlowASR/commit/526db4807dde87d9c258eafd7f6be09d2c0fcf6e))

- Wordpiece whitespace libri
  ([`618cf83`](https://github.com/TensorSpeech/TensorFlowASR/commit/618cf83182ed5f0321bf2ac33122d139041d28b6))

- Wordpiece whitespace metadata
  ([`0a9d339`](https://github.com/TensorSpeech/TensorFlowASR/commit/0a9d3394a92d7cf4dda79256b74cf7d6bf4d3b5c))

- Wp whitespace libri
  ([`826d0a8`](https://github.com/TensorSpeech/TensorFlowASR/commit/826d0a803cad6bb220490ea212323edb0ad88ad6))

- **deps**: Bump black from 23.7.0 to 24.3.0
  ([`4ea8283`](https://github.com/TensorSpeech/TensorFlowASR/commit/4ea82837a919c3caf1ab132d28e35b9a3c13f109))

- **deps**: Bump jinja2 from 3.1.2 to 3.1.3
  ([`bc73439`](https://github.com/TensorSpeech/TensorFlowASR/commit/bc73439a89199340278d4df999bc29aa21f45784))

- **deps**: Bump tqdm from 4.66.1 to 4.66.3
  ([`a5757fa`](https://github.com/TensorSpeech/TensorFlowASR/commit/a5757faa7166cb474d9932580065b4eaa0046476))

### Features

- Add blurpool layers
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add causal padding support for conv2d
  ([`607d9c2`](https://github.com/TensorSpeech/TensorFlowASR/commit/607d9c2a4ab997e815a81bca710460a1703e229b))

- Add compute output shape explicitly
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add conformer small no decay result
  ([`8c3b653`](https://github.com/TensorSpeech/TensorFlowASR/commit/8c3b6539b56f98f342113c24762dc4d927f4b1bf))

- Add conv1d subsampling
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add dataset enabled flag
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add decoder gaussian weight noise
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add feature extraction layer, fix models, update dataset
  ([`08ca341`](https://github.com/TensorSpeech/TensorFlowASR/commit/08ca3418c2afe9c58405d3d57f67f70221813064))

- Add gauss noise to augmentation, remove layer gauss noise, update cache order
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add gaussian gradient noise
  ([`d163774`](https://github.com/TensorSpeech/TensorFlowASR/commit/d16377482fd49c45266c82a5bc9202eefc79c693))

- Add get/set states of mha memory
  ([`057963c`](https://github.com/TensorSpeech/TensorFlowASR/commit/057963c477d440adf79cf259c0d3ba98e97dd332))

- Add jinja2 template when loading yaml
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add memory in relmha
  ([`e144b2a`](https://github.com/TensorSpeech/TensorFlowASR/commit/e144b2acb45caef3e9574003e7ee011c18e1397a))

- Add memory to conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add one_hot label encoder
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add options for checkpoints
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add pretrained to learning config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add recognize from signals for transducer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add rezero conformer
  ([`85f8757`](https://github.com/TensorSpeech/TensorFlowASR/commit/85f87574a941e2e352f956f27f4aeba9fd5936a9))

- Add rnn to decoder conformer ctc
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add rnnt loss naive
  ([`0ec90b8`](https://github.com/TensorSpeech/TensorFlowASR/commit/0ec90b890eb339ecc0312f3a56f55e545b554dc4))

- Add script to gen vocab wordpiece
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add self attention mask to conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add variational noise to decoder
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add vgg blurpool subsampling
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add wer, cer to validation stage
  ([`01f9dcc`](https://github.com/TensorSpeech/TensorFlowASR/commit/01f9dcc0b2fe2f8fb81c73218f78023f8a970327))

- Add wordpiece featurizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Add wordpiece metadata
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Allow space in wordpiece featurizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Fast sentencepiece tokenizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Gradient accumulation
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Introduce gaussian noise to inputs of conformer + contextnet + rnn_transducer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Memory caching training
  ([`4610156`](https://github.com/TensorSpeech/TensorFlowASR/commit/4610156bc1f9502495da167be285899a2dddb1f2))

- Memory conformer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Memory mha
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Refactor imports to support tf2.13
  ([`126b45a`](https://github.com/TensorSpeech/TensorFlowASR/commit/126b45a81594c41c8c0413361eef68c3951c3c77))

- Refactor tokenizer structure, update testing
  ([`41e4326`](https://github.com/TensorSpeech/TensorFlowASR/commit/41e4326ed9d05d0af6d3d06ab53ab99e64e1cb8a))

- Register keras custom objects
  ([`dbf43cf`](https://github.com/TensorSpeech/TensorFlowASR/commit/dbf43cfe36bd5c6cd78238366629db7023625596))

- Relmha memory
  ([`2f9cb22`](https://github.com/TensorSpeech/TensorFlowASR/commit/2f9cb2288daf8866fc0713c6b862ce046a45f282))

- Remove redundant scripts, add prepare vocab and metadata script
  ([`5326f40`](https://github.com/TensorSpeech/TensorFlowASR/commit/5326f401f7f1fc8cda554f6de925d4e8aaabad41))

- Remove use_tf, support keep whitespace for text features, fix transformer
  ([`f7b0d40`](https://github.com/TensorSpeech/TensorFlowASR/commit/f7b0d40bcb01b458123fd2bfce98d8f459a8fbdf))

- Rename params of gaussian gradient noise
  ([`7229dd5`](https://github.com/TensorSpeech/TensorFlowASR/commit/7229dd5fb63a1c93e2e6b5f25311544ba72256ec))

- Transformer transducer
  ([`96502f5`](https://github.com/TensorSpeech/TensorFlowASR/commit/96502f5340b1151c3af63345d3c73becb31c9490))

- Update memory mha
  ([`6025032`](https://github.com/TensorSpeech/TensorFlowASR/commit/602503288fea18581f4584eb72c22182bd1dc930))

- Update models, recognitions
  ([`927e0a1`](https://github.com/TensorSpeech/TensorFlowASR/commit/927e0a1a7d19f183d1e397c832d382ad3c7ee39f))

- Update results and scripts
  ([`944dd1e`](https://github.com/TensorSpeech/TensorFlowASR/commit/944dd1e3d534cadb516df2ee28686f5f416f970d))

- Update rnnt naive loss
  ([`ed67084`](https://github.com/TensorSpeech/TensorFlowASR/commit/ed67084598724c511a6f7919000f6ea25eb3b666))

- Update training tutorial
  ([`8f7cc5e`](https://github.com/TensorSpeech/TensorFlowASR/commit/8f7cc5e9c90a7991d27bb4dbec7153aff5bd0e23))

- Update wordpiece featurizer
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- Use default compile loss to handle missing regloss
  ([`a59891b`](https://github.com/TensorSpeech/TensorFlowASR/commit/a59891b5f8223e00e0ca5cd17f883ec2e6b59474))

- Use FastWordPieceTokenizer and update decoder config
  ([`10b3619`](https://github.com/TensorSpeech/TensorFlowASR/commit/10b3619402fdea9debe3b3b53276846ae1658666))

- **conformer**: Add option depthwise_as_groupwise
  ([`442f32c`](https://github.com/TensorSpeech/TensorFlowASR/commit/442f32c808cb8c62b0f4c761a50c6f831f272c62))


## v1.0.3 (2022-03-12)

### Bug Fixes

- Add preprocess paths to running config
  ([`aca16e4`](https://github.com/TensorSpeech/TensorFlowASR/commit/aca16e49bd1423ee2e1639b11e6683eda508dc54))

- Refactor using helpers
  ([`bdb52f7`](https://github.com/TensorSpeech/TensorFlowASR/commit/bdb52f722233cfa066a0ea95e34d58b4b45ce275))

- Update inference scripts for conformer
  ([`5801018`](https://github.com/TensorSpeech/TensorFlowASR/commit/580101899d1a58f92aa0f7c526ba2e9f94fbc2db))

- Update inference scripts for jasper + ds2 + rnn transducer
  ([`f54de55`](https://github.com/TensorSpeech/TensorFlowASR/commit/f54de55d72dc3ee2e0c72ea5d3bf716cc9078916))


## v1.0.2 (2021-11-07)

### Bug Fixes

- Refactor imports
  ([`fed73be`](https://github.com/TensorSpeech/TensorFlowASR/commit/fed73be6b5638e05c764e8872a6c2a8f0ede5e8a))

- Update saved model
  ([`c87fca6`](https://github.com/TensorSpeech/TensorFlowASR/commit/c87fca6aeb64107c4ea3fced0138e3b694f6c5f0))

- **ctc-decoding**: Add tokens support for subwords
  ([`501be44`](https://github.com/TensorSpeech/TensorFlowASR/commit/501be44ce55897107890ebfd9232d3224f0ecf2a))

- **example**: Update example configs
  ([`fec1051`](https://github.com/TensorSpeech/TensorFlowASR/commit/fec1051f813a1a1478ded997f00452028e39a302))

- **model**: Add comments on embedding
  ([`196c68b`](https://github.com/TensorSpeech/TensorFlowASR/commit/196c68b2b1b752250309a0b8f3dc8fb1baf07f5d))

- **model**: Incorrect metrics
  ([`2a40a6c`](https://github.com/TensorSpeech/TensorFlowASR/commit/2a40a6ca4aa6c02ddfa37dc59ac431b54f77b6de))

- **model**: Support saved model and tflite conversion
  ([`e318348`](https://github.com/TensorSpeech/TensorFlowASR/commit/e3183482104aa161d404609a7de879ae7a4c1fa9))

- **readme**: Support tensorflow >= 2.5.1
  ([`002ce43`](https://github.com/TensorSpeech/TensorFlowASR/commit/002ce43d2e0c1d65fd2f28f79bf1bb4c8dca57ac))

- **req**: Support tensorflow >= 2.5.1
  ([`0c9e22d`](https://github.com/TensorSpeech/TensorFlowASR/commit/0c9e22d5c67998fc3d422998fa55776b59b30b15))

- **req**: Update reqs structure
  ([`ba6ab51`](https://github.com/TensorSpeech/TensorFlowASR/commit/ba6ab512fa8d29a904c010ab6788ee3bfeaf3a0f))


## v1.0.1 (2021-05-16)


## v1.0.0 (2021-04-18)


## v0.8.3 (2021-04-10)


## v0.8.2 (2021-04-06)


## v0.8.1 (2021-03-18)


## v0.8.0 (2021-03-10)


## v0.7.8 (2021-02-24)


## v0.7.7 (2021-02-21)


## v0.7.6 (2021-02-20)


## v0.7.5 (2021-02-16)


## v0.7.4 (2021-02-13)


## v0.7.3 (2021-02-12)


## v0.7.2 (2021-02-07)


## v0.7.1 (2021-01-31)


## v0.7.0 (2021-01-24)


## v0.6.4 (2021-01-16)

### Bug Fixes

- Tensor to list for test evaluation
  ([`89756ee`](https://github.com/TensorSpeech/TensorFlowASR/commit/89756eea6c051a65434d13c05ad67c81f7fe16c3))


## v0.6.3 (2021-01-06)


## v0.6.2 (2020-12-27)


## v0.6.1 (2020-12-27)


## v0.6.0 (2020-12-27)


## v0.5.5 (2020-12-25)


## v0.5.4 (2020-12-24)


## v0.5.3 (2020-12-20)


## v0.5.2 (2020-12-20)


## v0.5.1 (2020-12-19)


## v0.5.0 (2020-12-17)


## v0.4.5 (2020-12-15)


## v0.4.4 (2020-12-15)


## v0.4.3 (2020-12-13)


## v0.4.2 (2020-12-11)


## v0.4.1 (2020-12-10)


## v0.4.0 (2020-12-06)


## v0.3.2 (2020-12-05)


## v0.3.1 (2020-11-22)


## v0.3.0 (2020-11-14)


## v0.2.10 (2020-11-03)


## v0.2.9 (2020-10-31)


## v0.2.8 (2020-10-20)


## v0.2.7 (2020-10-18)


## v0.2.6 (2020-10-17)


## v0.2.5 (2020-10-15)

- Initial Release

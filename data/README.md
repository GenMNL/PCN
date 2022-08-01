# データセットについて

---

## Shapenet completion dataset
- 普通の検証では全てのカテゴリーで検証しているが，今回は椅子またはテーブルに合わせた検証をしたいため，そのようにdata.pyを書いている．

## modelnet40
- pytorchのpcnで使われていたライブラリ
- しかし，読み込んでいたデータセットはclassificationのデータセットだったため，completion用ではないと思われる．
- pytorchのpcnで行われていたのはどちらかというとcompletionではなくupsamplingだと思われる．
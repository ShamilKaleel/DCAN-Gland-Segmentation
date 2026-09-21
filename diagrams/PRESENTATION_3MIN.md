# 3-Minute Presentation Script — Explaining the Two Diagrams

Read this aloud at a calm pace; it lands at about **3 minutes**.
`[…]` are pointing cues — physically point at that part of the diagram as you say it.

**The one line to memorize (your anchor):**
> *DCAN learns the gland **and** its outline at the same time, so it can separate glands that touch.*

---

## Part 1 · The Model — `01_model_architecture.drawio`  (~90 seconds)

> "First, let me explain the model.
>
> The problem we are solving: in these tissue images, **glands often touch each other**, and a
> normal model merges two touching glands into one blob. DCAN is designed to avoid that.
>
> On the left, the image goes into a **shared encoder** — these blue boxes." *[run your finger
> left to right along the encoder row]* "This is a standard CNN that pulls out features, getting
> deeper and more abstract as we move right.
>
> Now here is the key idea of DCAN: the model splits into **two branches**." *[point to the top
> row, then the bottom row]*
>
> "The **top branch** predicts the **gland body** — where the gland is." *[point to p_obj on the
> right]*
> "The **bottom branch** — this is DCAN's special addition — predicts the **gland edges**, the
> outlines." *[point to p_cnt]*
>
> "Each branch doesn't just use the final features. It taps the network at **three different
> depths**" *[trace the three dashed lines feeding into the heads]* "so it sees both small detail
> and the big picture, then **adds them together**, upscales back to full size, and produces a
> probability map.
>
> So we get two outputs: a **gland-body map** and a **gland-edge map**.
>
> Finally, on the right, the **fusion rule** combines them:" *[point to the red Fusion box]*
> "keep the gland body, but **cut along the predicted edges**. That cut is what separates two
> glands that are touching. That is the whole trick of DCAN."

---

## Part 2 · The Pipeline — `02_pipeline.drawio`  (~60 seconds)

> "This second diagram shows the full process from start to finish." *[trace the boxes following
> the arrows]*
>
> "We **input** the histology image. We **preprocess** it — build the training targets and apply
> data augmentation like flips, rotations and color jitter. Then we **train** the DCAN model with
> its two branches.
>
> At test time, we run **inference** — we slide the model over the image in overlapping tiles to
> get the body map and the edge map. We **post-process** them: apply the fusion rule, then simple
> morphology — opening and filling holes — to get clean, **separated gland instances**.
>
> Lastly we **evaluate** with the three official metrics: detection **F1**, object **Dice**, and
> object **Hausdorff**."

---

## Closing line (~10 seconds)

> "So in short:" *[point back to the two branches in Diagram 1]* "by learning the gland body and
> its outline together, DCAN can pull apart glands that touch — which simple models cannot do."

---

## Delivery tips
- **Point as you talk** — move your hand/cursor so the lecturer's eyes follow the data flow.
- **Spend most time on the two branches** — that is the only truly novel part; everything else is standard.
- **Pause for one beat when you switch diagrams** ("That's the model — now the full pipeline.").
- **Use plain words:** "gland body / gland edges", not "object/contour logits".
- **Timing:** Diagram 1 ≈ half your time, Diagram 2 ≈ one-third, opening + closing the rest. If you run long, trim the pipeline (it's the simple one) — never the two-branch explanation.
- **Practice once with a timer**, aiming for ~2:45 so you have room for a question.

---

## Per-box cheat sheet (Diagram 1 — if the lecturer asks about a specific box)

**Reading the labels:** `↓/8` = image shrunk to 1/8 size · `256` = 256 feature channels (each channel = one feature detector, e.g. edges, stain, round shapes). Going deeper: picture gets **smaller**, features get **richer**.

### Encoder (blue, shared feature extractor — a VGG-16 pre-trained on ImageNet)
| Box | Say this |
|---|---|
| **Input 480×480×3** | The image patch; "3" = Red/Green/Blue channels. |
| **enc1 (↓/2, 64)** | First conv layers — basic edges & color. Half size, 64 features. |
| **enc2 (↓/4, 128)** | Simple shapes & textures. Quarter size, 128 features. |
| **enc3 (↓/8, 256)** | Gland-part patterns. 1/8 size, 256 features. **Tapped by the heads.** |
| **enc4 (/8, 512)** | Higher-level shapes. Still 1/8, 512 features. **Tapped.** |
| **enc5 (/8, 512)** | Big-context features; held at 1/8 by the atrous trick. |
| **fc6 (/8, 1024)** | Deepest, widest view — 1024 features. **Tapped.** (purple) |

**What's inside the encoder:** stacks of **convolutions** (sliding pattern-detectors) + **pooling** (which halves the size). Normal VGG shrinks to 1/32 (too blurry for sharp edges). DCAN **stops at 1/8** by setting the last pooling layers to **stride 1**, and uses **atrous / dilated convolution** (the "dilation 2 / 12" notes) — *spreading the filter's fingers apart to see a wide area without shrinking the image*. This keeps gland **edges sharp**.

### Object branch (green, top) and Contour branch (orange, bottom) — same 4 boxes
| Box | Say this |
|---|---|
| **3 multi-level heads (enc3 · enc4 · fc6)** | Three mini-classifiers reading **three depths**, so the branch sees fine detail *and* big-picture context. |
| **⊕ sum-fusion @ /8** | **Add** the three predictions into one map (at 1/8 size). |
| **upsample** | **Enlarge** it back to full 480×480. |
| **softmax** | Turn scores into **probabilities** (0–1). |
| **p_obj / p_cnt** | Final maps: **p_obj** = "how likely gland body", **p_cnt** = "how likely gland edge". |

*The only difference between the two branches: the top is trained on the filled gland, the bottom on the outline.*

### Fusion (red, right)
| Box | Say this |
|---|---|
| **Fusion rule → instance map** | A pixel is a gland if **object prob is high AND contour prob is low**. The edges act as **cut lines** that split touching glands into separate objects. |

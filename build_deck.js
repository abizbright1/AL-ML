// build_deck.js — class presentation, PathologyLoss
// Every figure is a real BraTS 2021 run from this repository. Numbers come
// from results/comparison/phase2_seed1 unless the slide says otherwise.
const pptxgen = require('pptxgenjs');
const path = require('path');
const { applyTheme } = require('/root/.claude/skills/synced/75eb06ac-5962-4d6d-8469-acfbaf310754_05ac8f3e-0b33-4f04-bc53-bd1bf3e5de7c/pptx/scripts/apply_theme.js');

// Restrained palette: slate and a single teal, with amber reserved for the
// enhancing tumour and nothing else. Rust marks cost, never decoration.
const THEME = {
  name: 'Reading Room',
  headFontFace: 'Cambria',
  bodyFontFace: 'Calibri',
  colors: {
    dk1: '101E26', lt1: 'FFFFFF',
    dk2: '16303D', lt2: 'F3F6F7',
    accent1: '1C7293', accent2: 'C97B16', accent3: '4E7A6B',
    accent4: '8C3B2F', accent5: '5A7484', accent6: 'B4C6CE',
    hlink: '1C7293', folHlink: '4E7A6B',
  },
};
const K = THEME.colors;
const A = (f) => path.join(__dirname, 'deck_assets', f);
const F = (f) => path.join(__dirname, 'results/comparison/phase2_seed1/paper_figures', f);

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.author = 'Abika Osinachi Bright';
pres.title = 'PathologyLoss';
const C = pres.SchemeColor;

const FOOT = 'BraTS 2021 · 100 patients · 80/20 by patient · all figures from real runs';

pres.defineSlideMaster({
  title: 'DARK', background: { color: K.dk1 },
  objects: [
    { placeholder: { options: { name: 'eyebrow', type: 'body', x: 1.1, y: 1.85, w: 11.1, h: 0.35,
        fontSize: 12, color: K.accent2, charSpacing: 4, bold: true }, text: ' ' } },
    { placeholder: { options: { name: 'title', type: 'title', x: 1.1, y: 2.3, w: 11.1, h: 1.8,
        fontSize: 42, bold: true, color: C.background1, valign: 'top' }, text: ' ' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 1.1, y: 4.3, w: 9.9, h: 1.9,
        fontSize: 16, color: K.accent6, lineSpacing: 26 }, text: ' ' } },
  ],
});

pres.defineSlideMaster({
  title: 'STATEMENT', background: { color: K.dk2 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 1.1, y: 2.0, w: 11.1, h: 2.5,
        fontSize: 38, bold: true, color: C.background1, valign: 'middle' }, text: ' ' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 1.1, y: 4.75, w: 10.4, h: 1.4,
        fontSize: 16, color: K.accent6, lineSpacing: 26 }, text: ' ' } },
  ],
});

pres.defineSlideMaster({
  title: 'CONTENT', background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 0.52, w: 11.7, h: 0.8,
        fontSize: 30, bold: true, color: K.dk2, valign: 'top' }, text: ' ' } },
    { text: { text: FOOT, options: { x: 0.8, y: 6.98, w: 10.4, h: 0.28, fontSize: 8.5,
        color: K.accent5, isTextBox: true } } },
  ],
  slideNumber: { x: 12.2, y: 6.98, w: 0.6, fontSize: 8.5, color: K.accent5 },
});

// ── helpers ────────────────────────────────────────────────────────────────
const t = (s, txt, o) => s.addText(txt, Object.assign({ isTextBox: true, margin: 0 }, o));

function lede(s, txt, w = 11.7) {
  t(s, txt, { x: 0.8, y: 1.42, w, h: 0.62, fontSize: 15, color: K.dk1, lineSpacing: 22 });
}
// Height is derived from the file's real pixel dimensions. Passing both w and
// h by hand is how an image ends up stretched, and a squashed brain scan in a
// medical talk is a credibility problem, not a cosmetic one.
const sizeOf = require('image-size').default || require('image-size');
function plateW(s, img, x, y, w, cap) {
  const d = sizeOf(require('fs').readFileSync(img));
  plate(s, img, x, y, w, w * d.height / d.width, cap);
}
function plate(s, img, x, y, w, h, cap) {
  s.addShape(pres.ShapeType.roundRect, { x: x - 0.1, y: y - 0.1, w: w + 0.2, h: h + 0.2,
    rectRadius: 0.04, fill: { color: 'FFFFFF' }, line: { width: 0 },
    shadow: { type: 'outer', blur: 12, offset: 2, angle: 90, color: 'C3D0D6', opacity: 0.55 } });
  s.addImage({ path: img, x, y, w, h });
  if (cap) t(s, cap, { x, y: y + h + 0.16, w, h: 0.5, fontSize: 10.5,
    color: K.accent5, lineSpacing: 15 });
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.05,
    fill: { color: fill || K.lt2 }, line: { width: 0 } });
}
function stat(s, x, y, w, big, label, sub, col) {
  card(s, x, y, w, 1.85);
  t(s, big, { x: x + 0.24, y: y + 0.18, w: w - 0.48, h: 0.8, fontSize: 38, bold: true, color: col });
  t(s, label, { x: x + 0.24, y: y + 0.98, w: w - 0.48, h: 0.32, fontSize: 12.5, bold: true, color: K.dk2 });
  t(s, sub, { x: x + 0.24, y: y + 1.30, w: w - 0.48, h: 0.45, fontSize: 10, color: K.accent5, lineSpacing: 14 });
}
function keyline(s, txt, col) {
  t(s, txt, { x: 0.8, y: 6.25, w: 11.7, h: 0.5, fontSize: 15, bold: true, italic: true,
    color: col || K.dk2 });
}

// ══════════════════════════════════════════════════════════════════════════
pres.addSection({ title: 'Opening' });

let s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Opening' });
s.addText('LOUISIANA TECH UNIVERSITY  ·  MEDICAL IMAGE ANALYSIS', { placeholder: 'eyebrow' });
s.addText([{ text: 'Teaching a network ', options: { color: 'FFFFFF' } },
           { text: 'where to look', options: { color: K.accent2, italic: true } }],
          { placeholder: 'title' });
s.addText('A region-weighted training objective for brain-tumour MRI restoration, '
  + 'benchmarked against thirteen published alternatives — and an honest account of '
  + 'what it does and does not buy you.', { placeholder: 'body' });
s.addNotes('One idea: change what the network is graded on, not the network. '
  + 'Every image in this deck is from a real BraTS run on my own machine.');

// 2 — THE HOOK, two real patients
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Opening' });
s.addText('The same disease, sixteen times smaller', { placeholder: 'title' });
lede(s, 'Two patients from the dataset. Left, the enhancing tumour fills 10.4% of the slice. '
  + 'Right, it fills 0.64%. A pixel-averaged loss treats both identically.');
plateW(s, A('regions_large.png'), 0.8, 2.3, 5.55,
  'Patient A — WT 37.3%, TC 12.5%, ET 10.4% of the slice');
plateW(s, A('regions_small.png'), 6.95, 2.3, 5.55,
  'Patient B — WT 8.3%, TC 0.64%, ET 0.64% of the slice');
card(s, 0.8, 4.55, 11.7, 1.5, K.dk2);
t(s, 'The enhancing tumour drives staging, surgical planning and response assessment. '
  + 'It is also, routinely, under one percent of the image — so a network trained to '
  + 'minimise average pixel error has almost no incentive to preserve it.',
  { x: 1.15, y: 4.85, w: 11.0, h: 0.95, fontSize: 15, color: 'FFFFFF', lineSpacing: 23 });
s.addNotes('The hook. Pause on the two percentages before explaining anything. '
  + 'Both figures are real slices from the validation set.');

pres.addSection({ title: 'Problem' });

// 3 — the corruption, real
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Problem' });
s.addText('A faster scan is a noisier scan', { placeholder: 'title' });
lede(s, 'MRI trades time for quality. Children, patients in pain and anyone who cannot '
  + 'lie still do not get a ten-minute acquisition. We simulate that trade with Rician '
  + 'noise at sigma 0.08.', 7.0);
plateW(s, A('noise.png'), 0.8, 2.1, 5.6,
  'Clean, corrupted, and the noise field itself. T1ce falls to 22.3 dB, FLAIR to 21.9 dB.');
card(s, 7.0, 2.1, 5.5, 2.6);
t(s, 'What is lost first', { x: 7.3, y: 2.35, w: 4.9, h: 0.35, fontSize: 14, bold: true, color: K.dk2 });
t(s, 'Fine structure at the tumour margin — exactly the boundary a radiologist traces '
  + 'to decide whether disease is progressing.',
  { x: 7.3, y: 2.78, w: 4.9, h: 1.1, fontSize: 13, color: K.dk1, lineSpacing: 20 });
t(s, 'The denoiser sits between the scanner and the diagnosis. Whatever it destroys, '
  + 'no later step recovers.', { x: 7.3, y: 3.9, w: 4.9, h: 0.7, fontSize: 13,
    italic: true, color: K.accent4, lineSpacing: 19 });
s.addNotes('Real figure from the run. Point at the noise panel on the right — '
  + 'that is what the network has to undo.');

// 4 — PROBLEM STATEMENT
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Problem' });
s.addText('The network is graded on the wrong thing', { placeholder: 'title' });
lede(s, 'Training minimises the average error over every pixel. That average belongs to '
  + 'the healthy brain.');
card(s, 0.8, 2.25, 5.55, 3.75);
t(s, 'WHAT THE LOSS REWARDS', { x: 1.1, y: 2.5, w: 5.0, h: 0.32, fontSize: 11.5, bold: true,
  color: K.accent5, charSpacing: 2 });
t(s, '99.4%', { x: 1.1, y: 2.95, w: 5.0, h: 1.0, fontSize: 62, bold: true, color: K.accent5 });
t(s, 'of pixels drive the gradient', { x: 1.1, y: 3.95, w: 5.0, h: 0.35, fontSize: 12.5, color: K.accent5 });
t(s, 'A smooth reconstruction of healthy tissue. Score it on PSNR and it looks excellent.',
  { x: 1.1, y: 4.45, w: 5.0, h: 1.1, fontSize: 13.5, color: K.dk1, lineSpacing: 20 });

card(s, 6.95, 2.25, 5.55, 3.75, K.dk2);
t(s, 'WHAT THE CLINIC NEEDS', { x: 7.25, y: 2.5, w: 5.0, h: 0.32, fontSize: 11.5, bold: true,
  color: K.accent2, charSpacing: 2 });
t(s, '0.6%', { x: 7.25, y: 2.95, w: 5.0, h: 1.0, fontSize: 62, bold: true, color: K.accent2 });
t(s, 'of pixels carry the decision', { x: 7.25, y: 3.95, w: 5.0, h: 0.35, fontSize: 12.5, color: K.accent6 });
t(s, 'A faithful enhancing-tumour margin. The network can smear it and barely move the average.',
  { x: 7.25, y: 4.45, w: 5.0, h: 1.1, fontSize: 13.5, color: 'FFFFFF', lineSpacing: 20 });
keyline(s, 'The metric we optimise is not the thing we care about.', K.accent4);
s.addNotes('The thesis. Say the closing line slowly; everything after this follows from it.');

pres.addSection({ title: 'Method' });

s = pres.addSlide({ masterName: 'STATEMENT', sectionTitle: 'Method' });
s.addText('Change the grading rubric, not the student', { placeholder: 'title' });
s.addText('The architecture stays byte-for-byte identical to its control. Only the loss '
  + 'function changes — so any difference in the result belongs to the objective and '
  + 'nothing else.', { placeholder: 'body' });
s.addNotes('The scope decision. It is what makes this a loss study rather than an '
  + 'architecture study, and it is why the comparison is interpretable.');

// 6 — the method with the real weight map
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Method' });
s.addText('PathologyLoss', { placeholder: 'title' });
card(s, 0.8, 1.42, 7.0, 1.2, K.dk1);
t(s, 'L  =  L₁  +  Σ  wᵣ · ‖ Mᵣ ⊙ (X̂ − X) ‖₁ / |Mᵣ|',
  { x: 1.05, y: 1.56, w: 6.5, h: 0.55, fontSize: 19, bold: true, color: 'FFFFFF', fontFace: 'Cambria' });
t(s, 'global term   +   one size-normalised penalty per tumour region',
  { x: 1.05, y: 2.14, w: 6.5, h: 0.32, fontSize: 11, color: K.accent6 });
[['Errors inside the enhancing tumour count ×3, the core ×2, the rest ×1.', K.accent2],
 ['Each region is divided by its own size, so a small region is not diluted by being small.', K.accent1],
 ['The mask is used only in training. At inference the network sees the noisy image alone.', K.accent3],
].forEach(([txt, col], i) => {
  const y = 2.95 + i * 0.86;
  s.addShape(pres.ShapeType.ellipse, { x: 0.84, y: y + 0.05, w: 0.26, h: 0.26,
    fill: { color: col }, line: { width: 0 } });
  t(s, txt, { x: 1.28, y, w: 6.5, h: 0.78, fontSize: 13.5, color: K.dk1, lineSpacing: 20 });
});
plateW(s, A('weightmap.png'), 8.2, 2.5, 4.3,
  'The weight map on a real slice: red ×3, blue ×1. The entire intervention.');
s.addNotes('The equation, the three properties, and the real weight map. '
  + 'Note how little of the field is red — that is the imbalance the loss corrects.');

pres.addSection({ title: 'Experiment' });

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Experiment' });
s.addText('How it was tested', { placeholder: 'title' });
[['Split by patient', '100 subjects, 80 train / 20 held out. No patient appears in both — a leak here inflates every number downstream, and an earlier version of this study had one.'],
 ['Corrupt', 'Rician noise, sigma 0.08, applied in the image domain to all four modalities.'],
 ['Freeze the judge', 'A U-Net learns to outline tumours from clean images, then is locked. Every objective is scored by the identical network, so the ruler cannot drift.'],
 ['Train and compare', 'Same backbone, same data, same optimiser, same epochs. Only the loss differs.'],
].forEach(([h, d], i) => {
  const y = 1.52 + i * 1.3;
  card(s, 0.8, y, 11.7, 1.12);
  t(s, String(i + 1), { x: 1.1, y: y + 0.3, w: 0.4, h: 0.5, fontSize: 24, bold: true, color: K.accent6 });
  t(s, h, { x: 1.7, y: y + 0.22, w: 2.6, h: 0.4, fontSize: 15, bold: true, color: K.dk2 });
  t(s, d, { x: 4.4, y: y + 0.2, w: 7.8, h: 0.8, fontSize: 12.5, color: K.dk1, lineSpacing: 18 });
});
s.addNotes('The frozen judge is the methodological spine. Without it a Dice difference '
  + 'could be the segmentor changing rather than the images.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Experiment' });
s.addText('Benchmarked against published objectives', { placeholder: 'title' });
lede(s, 'An earlier version of this study compared PathologyLoss only against plain L₁ — '
  + 'our loss against nothing. That is a control, not a benchmark.');
[['REGION-AWARE RIVALS', ['FlatROI — binary tumour weighting', 'BraSyn masked L₁ (2023)', 'ROIRecNet-inspired (2019)'], K.accent2],
 ['CONVENTIONAL STRENGTH', ['L₂, SSIM, MS-SSIM', 'Mix (Zhao 2017)', 'Total variation, VGG perceptual'], K.accent1],
 ['FREQUENCY & TASK-AWARE', ['BraSyn frequency split', 'DAGAN-style (Yang 2017)', 'LIDnet-inspired task feedback'], K.accent3],
].forEach(([h, items, col], i) => {
  const x = 0.8 + i * 4.0;
  card(s, x, 2.25, 3.7, 2.95);
  s.addShape(pres.ShapeType.rect, { x: x + 0.3, y: 2.55, w: 0.26, h: 0.05,
    fill: { color: col }, line: { width: 0 } });
  t(s, h, { x: x + 0.3, y: 2.68, w: 3.1, h: 0.32, fontSize: 11, bold: true, color: col, charSpacing: 1.5 });
  s.addText(items.map((x2, j) => ({ text: x2, options: { bullet: true, breakLine: j < items.length - 1 } })),
    { x: x + 0.3, y: 3.1, w: 3.1, h: 1.9, fontSize: 12, color: K.dk1,
      isTextBox: true, margin: 0, paraSpaceAfter: 8, lineSpacing: 18 });
});
card(s, 0.8, 5.45, 11.7, 1.0, K.dk2);
t(s, 'Plus two reference arms that cost no training at all and change how every other '
  + 'number reads: segment the noisy image, and segment the clean one.',
  { x: 1.15, y: 5.68, w: 11.0, h: 0.6, fontSize: 14, color: 'FFFFFF', lineSpacing: 20 });
s.addNotes('Be candid that the first version had no real baselines. That criticism is '
  + 'what produced this slide and most of the work behind it.');

pres.addSection({ title: 'Results' });

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
s.addText('First question: does denoising help the diagnosis at all?', { placeholder: 'title' });
lede(s, 'Nobody had measured this. Without it, every Dice score in the study floats free.');
stat(s, 0.8, 2.2, 3.6, '0.244', 'No restoration', 'segment the noisy image directly', K.accent5);
stat(s, 4.85, 2.2, 3.6, '0.514', 'After denoising', 'plain L₁ objective', K.accent1);
stat(s, 8.9, 2.2, 3.6, '0.617', 'Clean image', 'the ceiling — best achievable', K.accent3);
card(s, 0.8, 4.4, 11.7, 1.6);
t(s, 'Restoration recovers 72% of the achievable gap.',
  { x: 1.15, y: 4.62, w: 11.0, h: 0.4, fontSize: 16, bold: true, color: K.dk2 });
t(s, 'That leaves 0.103 of headroom. A loss function that wins +0.05 is claiming half of '
  + 'what remains — a far more meaningful statement than a Dice number quoted alone.',
  { x: 1.15, y: 5.05, w: 11.0, h: 0.8, fontSize: 14, color: K.dk1, lineSpacing: 21 });
s.addNotes('Dice_ET, 20 held-out patients. Floor and ceiling are what make everything '
  + 'after this interpretable.');

// 10 — the real denoised strip
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
s.addText('What each objective produces', { placeholder: 'title' });
lede(s, 'The same slice, restored by each objective, with its PSNR. By eye they are '
  + 'nearly indistinguishable — which is the problem with judging restoration by eye.');
plateW(s, A('denoised_strip.png'), 0.8, 2.6, 11.7);
card(s, 0.8, 5.35, 11.7, 1.05, K.dk2);
t(s, 'Every one of these looks acceptable. The differences that matter are not in the '
  + 'picture — they are in what a segmentor can still find afterwards.',
  { x: 1.15, y: 5.58, w: 11.0, h: 0.6, fontSize: 14, color: 'FFFFFF', lineSpacing: 20 });
s.addNotes('Real figure. Let the audience look for a difference and fail to find one. '
  + 'That failure is the argument for the next slide.');

// 11 — the real segmentation comparison
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
s.addText('What the segmentor finds afterwards', { placeholder: 'title' });
lede(s, 'Top row: the frozen segmentor on each restoration. Bottom: green where it found '
  + 'real enhancing tumour, red where it missed, blue where it invented tumour that is not there.');
plateW(s, A('seg_strip.png'), 1.35, 2.55, 10.6);
keyline(s, 'Uformer+PL finds the most real tumour — and also paints the largest false region.',
  K.accent4);
s.addNotes('The single most important image in the deck. PathologyLoss recovers more '
  + 'green AND more blue. That trade is invisible in any Dice table.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
s.addText('Every objective, both backbones', { placeholder: 'title' });
plateW(s, F('fig1_accuracy_by_loss.png'), 1.5, 1.45, 10.3);
t(s, 'Subject-level means with bootstrap 95% confidence intervals. Dotted line: no '
  + 'restoration. Dashed line: clean image.',
  { x: 0.8, y: 6.25, w: 11.7, h: 0.5, fontSize: 11.5, color: K.accent5 });
s.addNotes('The backbones disagree about which rival is best — but PathologyLoss is the '
  + 'only region-aware objective above plain L1 on both.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
s.addText('What the improvement is actually made of', { placeholder: 'title' });
plateW(s, F('fig5_decomposition_swinir.png'), 0.8, 1.6, 6.5);
card(s, 7.7, 1.6, 4.8, 4.06);
t(s, 'Verifying the code against the manuscript showed the published treatment also '
  + 'carried an SSIM term that its L₁ control did not.',
  { x: 8.0, y: 1.9, w: 4.2, h: 1.2, fontSize: 13.5, color: K.dk1, lineSpacing: 20 });
t(s, 'So the comparison was never isolating the region weighting.',
  { x: 8.0, y: 3.15, w: 4.2, h: 0.7, fontSize: 14, bold: true, color: K.accent4, lineSpacing: 20 });
t(s, 'The fix: a matched control carrying every non-pathology term, and the whole '
  + 'comparison re-run against that instead.',
  { x: 8.0, y: 3.95, w: 4.2, h: 1.2, fontSize: 13.5, color: K.dk1, lineSpacing: 20 });
keyline(s, 'A result you cannot attribute is not a result.');
s.addNotes('This slide is what makes the talk credible. Admitting the confound and '
  + 'showing the fix beats a clean result with no self-criticism.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
s.addText('Where the method genuinely wins', { placeholder: 'title' });
plateW(s, F('fig3_accuracy_vs_psnr_et.png'), 7.1, 1.5, 5.4);
t(s, 'The original claim was that PathologyLoss reconstructs the tumour more faithfully. '
  + 'The study never measured that — whole-image PSNR averages over healthy brain.',
  { x: 0.8, y: 1.5, w: 5.9, h: 1.2, fontSize: 13.5, color: K.dk1, lineSpacing: 21 });
t(s, 'So we measured PSNR inside the enhancing tumour.',
  { x: 0.8, y: 2.75, w: 5.9, h: 0.4, fontSize: 14, bold: true, color: K.dk2 });
[['Uformer', '30.32 dB'], ['SwinIR', '29.99 dB']].forEach(([bb, v], i) => {
  const y = 3.35 + i * 1.0;
  card(s, 0.8, y, 5.9, 0.85);
  t(s, bb, { x: 1.1, y: y + 0.06, w: 2.0, h: 0.32, fontSize: 12.5, bold: true, color: K.dk2 });
  t(s, v, { x: 1.1, y: y + 0.37, w: 2.0, h: 0.4, fontSize: 19, bold: true, color: K.accent2 });
  t(s, 'best of all objectives', { x: 3.2, y: y + 0.26, w: 3.3, h: 0.4, fontSize: 11.5,
    color: K.accent5, valign: 'middle' });
});
t(s, 'Highest tumour-region fidelity on both backbones — a claim confirmed by a '
  + 'measurement that did not previously exist.',
  { x: 0.8, y: 5.5, w: 5.9, h: 0.8, fontSize: 13, italic: true, color: K.dk2, lineSpacing: 19 });
s.addNotes('The positive finding. It only emerged because we added a region-restricted '
  + 'metric, which is itself part of the contribution.');

// 15 — error maps, real
s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Results' });
s.addText('Where the error actually lives', { placeholder: 'title' });
lede(s, 'Absolute error against the clean image, one shared colour scale. Brighter is worse.');
plateW(s, A('error_strip.png'), 0.8, 2.45, 11.7);
t(s, 'All objectives suppress noise comparably across the brain. The argument is not '
  + 'whether restoration works — it is what the residual error costs downstream.',
  { x: 0.8, y: 5.75, w: 11.7, h: 0.7, fontSize: 14, italic: true, color: K.dk2, lineSpacing: 20 });
s.addNotes('Reinforces slide 10: global fidelity is not where these methods differ.');

pres.addSection({ title: 'Reflection' });

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Reflection' });
s.addText('What I would not yet claim', { placeholder: 'title' });
[['One run, twenty patients', 'An earlier five-seed sweep had one seed in five reverse direction. This shows where things point, not how far.', K.accent4],
 ['It raises false positives', 'Across that sweep the enhancing-tumour false-positive rate rose in ten runs out of ten. Dice alone hides this.', K.accent4],
 ['The backbones disagree', 'The strongest rival on Uformer falls below plain L₁ on SwinIR. Portability is not settled.', K.accent5],
 ['Convergence is untested', 'Ten epochs here; a separate hundred-epoch run suggested the gap may close. That experiment is still to run.', K.accent5],
].forEach(([h, d, col], i) => {
  const x = 0.8 + (i % 2) * 5.95;
  const y = 1.55 + Math.floor(i / 2) * 2.4;
  card(s, x, y, 5.6, 2.1);
  s.addShape(pres.ShapeType.rect, { x: x + 0.3, y: y + 0.32, w: 0.26, h: 0.05,
    fill: { color: col }, line: { width: 0 } });
  t(s, h, { x: x + 0.3, y: y + 0.45, w: 5.0, h: 0.38, fontSize: 14.5, bold: true, color: K.dk2 });
  t(s, d, { x: x + 0.3, y: y + 0.92, w: 5.0, h: 1.1, fontSize: 12.5, color: K.dk1, lineSpacing: 19 });
});
s.addNotes('Say these out loud. A reviewer who finds them first is a far worse outcome '
  + 'than an author who names them.');

s = pres.addSlide({ masterName: 'CONTENT', sectionTitle: 'Reflection' });
s.addText('What happens next', { placeholder: 'title' });
[['Run to convergence', 'Fifty epochs on one seed, with the loss curves as evidence that the budget was sufficient.', '1'],
 ['Widen the cohort', 'Seed-to-seed variation traced to which patients land in the held-out set, not to training instability. More patients, not more seeds.', '2'],
 ['Report the trade honestly', 'Dice beside the false-positive rate and tumour-region PSNR, so sensitivity and specificity are both visible.', '3'],
].forEach(([h, d, n], i) => {
  const y = 1.6 + i * 1.65;
  card(s, 0.8, y, 11.7, 1.42);
  t(s, n, { x: 1.1, y: y + 0.42, w: 0.4, h: 0.55, fontSize: 26, bold: true, color: K.accent6 });
  t(s, h, { x: 1.75, y: y + 0.3, w: 3.1, h: 0.4, fontSize: 15, bold: true, color: K.dk2 });
  t(s, d, { x: 5.0, y: y + 0.28, w: 7.2, h: 0.9, fontSize: 13, color: K.dk1, lineSpacing: 19 });
});
s.addNotes('Three concrete next steps, each tied to something the data showed.');

s = pres.addSlide({ masterName: 'DARK', sectionTitle: 'Reflection' });
s.addText('WHAT THIS PROJECT TAUGHT ME', { placeholder: 'eyebrow' });
s.addText('A baseline is not a formality', { placeholder: 'title' });
s.addText('Almost everything that changed here came from asking what the result was being '
  + 'compared against. A floor and a ceiling turned an unanchored number into a meaningful '
  + 'one. A matched control revealed a confound. A region-restricted metric confirmed a '
  + 'claim the study had only asserted.', { placeholder: 'body' });
s.addNotes('Close on the methodological lesson rather than the effect size. It generalises '
  + 'beyond this project, and it is what the audience will remember.');

pres.writeFile({ fileName: 'PathologyLoss_Presentation.pptx' })
  .then(() => applyTheme('PathologyLoss_Presentation.pptx', THEME))
  .then(() => console.log('deck written + theme applied'))
  .catch((e) => { console.error('FAILED:', e.message); process.exit(1); });

# Show polish checklist

Owner: polish agent. Pass 1 at 13:30 (review only, design owned the files); pass 2 from 13:40 (design → polish handoff, b9b2119).
Bundle: r01 / w0002, 28 cues. `?cue=N` shows the state after cues 0..N-1, so screenshot N = after cue N-1.
Screenshots: `/tmp/claude-1000/-home-saturn-Documents-thinking-blufflab/4476e947-6ebf-4fa4-a7a3-5af4959bb1e1/scratchpad/polish/`
(`before-1080/`, `before-720/` = the handoff state; `after4-1080/`, `after5-720/` = after the fixes; `live/` = mid-animation frames captured by real keypresses over CDP).
Severity: **blocker** = visibly broken on stage; **major** = hurts reading from the back; **minor** = finish.

## Fixed
- [x] blocker, c12 evidence list ran off the stage. Design fixed it before handoff (last 3 rows + "and N earlier reveals").
- [x] blocker, c23 results table overflowed. Design fixed it before handoff (single-line CIs).
- [x] blocker, c06/c07 "Challenge!" was drawn over the claim quote, then (after design's move) over the Profiler's leftmost card. Now it sits on open felt left of the hand (`.seat.p1 .act`).
- [x] blocker, transient `renderStack is not defined` while design was mid-edit. Gone at handoff; every cue renders with no console errors at 1920×1080 and 1280×720.
- [x] major, c27 forge looked broken ("Forge,", "— vs Profiler —", "20, ,", a stretched chip). forge.json nests the sandbox result under `entry.forge_result`, and the scene read the top level. It now reads both levels: sandbox id, 60% (56–64%) vs Profiler 35%, 600/0/0, promoted yes, all from forge.json. Rows with no data are dropped instead of drawn as dashes.
- [x] major, c07/c10 faded old bubbles were 42% opacity, so the felt rim cut through "(voice: ElevenLabs)". They are now opaque, muted bubbles.
- [x] major, c10 freeze banner sat on the table rim. It now has a solid plate and sits higher.
- [x] major, c10→c11 table turn: the table stayed opaque while the inspector arrived, which gave a double exposure at about 0.6 s. The two now cross-fade cleanly.
- [x] major, c18→c19 flip: the surgery card was already face up when the keypress rendered. It now turns on screen, then the "bluff" verdict and right/wrong judges follow.
- [x] major, c07 the "Bluff" stamp landed before the cards turned. It is now delayed 0.5 s.
- [x] major, c18/c19 surgery 4th column misalignment. Fixed by design (fixed-height subtitle block).
- [x] minor, c02 the active-seat glow had a hard rectangular edge (Player 2). It now uses a `closest-side` gradient.
- [x] minor, c21 long con: the x ticks "1" and "2" collided (1 dropped), and the ◎ marker had no label (added to the legend).
- [x] minor, c24 decisions: "+3.1 pts" touched the dashed zero line. Labels near zero now offset to their own side.
- [x] minor, c22/c24 short scenes floated at the top. Their content is now centred under the title, and the split panels are bigger.
- [x] minor, c26 "forge sandbox —" in How it's built. Now shows the real sandbox id from forge.json (c245666f).
- [x] minor, headings and the close line use `text-wrap: balance` (no one-word widows). Negative numbers use a true minus (−0.51, −10.5 pts).
- [x] minor, the "telltale" role tag after Player 3 on the flip was about 14px. It is now 0.78em.

## Open (minor, not worth the risk before freeze)
- [ ] c13 dossier as of game 4, c11 inspector: bottom third of the stage is empty (content is short by nature).
- [ ] c23 results CI line is 18px at 1080 and about 12px at 720. Faint on a projector. (Ablation CI labels now 18px with brighter ink.)
- [ ] c18 "Provenance-blind agent" wraps to two lines (columns stay aligned).

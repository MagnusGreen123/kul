# JEPA v6–v8 — Treningsanalyse og eksperimentlogg

## Oversikt

JEPA v6 er den fjerde JEPA-versjonen (v4-v6) og den første som oppnår **positiv reward i Pong** med en latent world model uten pixel-decoder i treningsløkken. Runnen fullforte 500k steps over 20.6 timer (Wandb ID: `74uc7yum`).

Sluttresultat: peak reward **+11**, avsluttet på **-12** etter en destabilisering i siste 40k steps.

## v6-konfigurasjon (endringer fra v5)

v5 hadde to problemer: SIGReg-vekten var for lav (0.01) som lot representasjonene bli for strukturerte/korrelerte (sigreg=5.3), og pred_weight var for høy (5.0) som dominerte treningssignalet. v6 reverterte til mer balanserte verdier:

| Parameter      | v5    | v6    | Begrunnelse                         |
|----------------|-------|-------|-------------------------------------|
| sigreg_weight  | 0.01  | 0.1   | Sterkere regularisering, lavere sigreg-tap |
| pred_weight    | 5.0   | 1.0   | Balansere pred vs. andre tap        |

Alt annet var likt: embed_dim=256, pred_depth=4, horizon=15, critic_ema_decay=0.98, critic_grad_clip=10.

## Treningsforløp — tre faser

### Fase 1: Oppvarming og stabilisering (200k-280k)

Runnen ble resumert fra checkpoint step 204k (forrige run `80j05hma` krasjet ved 244k). Reward var -20/-21 gjennom hele denne fasen. Criticens value-estimater sank fra +3.5 til -2.5, og critic_loss falt til 0.025. World model-tapet stabiliserte seg: sigreg=1.6 (ned fra 5.3 i v5), pred=0.011.

Denne fasen representerer criticen som kalibrerer seg til den negative reward-distribusjonen.

### Fase 2: Stabil laeringsfase (280k-450k)

Den mest bemerkelsesverdige perioden. Reward forbedret seg jevnt og trutt:

```
Step    Reward   Value    Critic Loss  Critic Grad  Entropy
280k    -17      -1.48    0.025        0.54         1.48
304k    -13      -1.00    0.023        0.45         1.46
320k     -6      -0.59    0.013        0.37         1.45
356k     -6      -0.10    0.015        0.33         1.37
392k     -6      -0.26    0.017        0.33         1.36
416k     -2      -0.24    0.020        0.34         1.33
436k     +2      -0.45    0.030        0.39         1.37
448k     +9      -0.60    0.045        0.48         1.40
460k    +11      -0.64    0.076        0.65         1.48
```

Kjennetegn ved denne fasen:
- **Jevn reward-forbedring** fra -17 til +11 over ~180k steps
- **Stabile value-estimater**: holdt seg mellom -1.5 og -0.1, aldri langt fra faktiske returns
- **Lav critic loss**: 0.01-0.04 — criticen predikerte returns noyaktig
- **Lave gradienter**: critic_grad 0.3-0.65 — ingen plutselige korreksjoner
- **Naturlig entropy-reduksjon**: 1.48 til 1.33 — agenten utnyttet gradvis bedre strategier

Dette er den lengste stabile laeringskurven vi har sett med JEPA-arkitekturen. World model fungerte godt som grunnlag for imagination-basert actor-critic-trening.

### Fase 3: Kollaps ved regime-overgang (460k-500k)

Rett etter peak reward (+11 ved 460k) kollapset treningen:

```
Step    Reward   Value    Critic Loss  Critic Grad  Entropy
460k    +11      -0.64    0.076        0.65         1.48
468k     -2      -0.42    0.141        0.93         1.55
476k     +1      +0.26    0.267        1.47         1.60
484k     +2      +1.02    0.381        1.88         1.62
489k    -11      +1.32    0.390        1.94         1.63
501k    -12      +1.97    0.330        2.00         1.65
```

## Kollapsmekansime: Distributional shift i returns

Gjennom den stabile fasen var criticen kalibrert for **negative returns** (reward -20 til -6, value rundt -1 til 0). Nar agenten plutselig begynte a vinne (reward +9, +11), endret return-distribusjonen seg fundamentalt:

1. **Critic loss eksploderer 20x** (0.02 til 0.39): TD-errors blir store fordi target returns er langt fra det criticen predikerte
2. **Gradienter 6x opp** (0.3 til 2.0): Store TD-errors gir store gradientoppdateringer
3. **Value overestimering**: Value gar fra -0.6 til +2.0 mens faktisk reward faller tilbake til -12
4. **Policy kollapser**: Actoren optimiserer mot de overvurderte value-estimatene, endrer policyen dramatisk, og besøker states criticen ikke er trent pa

`critic_ema_decay=0.98` (tau=0.02) betyr at target-criticen oppdateres raskt — den "jager" den ustabile live-criticen istedenfor a stabilisere den. `critic_grad_clip=10` var aldri aktiv (gradienter nådde maks 2.0), sa den ga ingen beskyttelse.

## Sammenligning med RSSM v24

RSSM v24 (den opprinnelige Dreamer-implementasjonen) oppnådde reward +19 uten denne typen kollaps. Nøkkelforskjeller:

- RSSM bruker **stokastisk latent** (h+z) som gir naturlig regularisering av value-estimater
- RSSM hadde **lavere horizon** i tidlige versjoner
- JEPA's deterministiske predictor gir mer presise men også mer "sprø" rollouts — nar de er gale, er de systematisk gale i same retning

## v7-plan: Stabilisere critic ved regime-overganger

Tre endringer, alle rettet mot actor-critic-stabilitet. World model og encoder er uendret — de fungerte godt i v6.

### Endring 1: Tregere target critic (critic_ema_decay: 0.98 → 0.995)

**Begrunnelse:** Target-criticen oppdateres med tau = 1 - decay. Ved 0.98 er tau=0.02 (2% av live critic per steg). Ved 0.995 er tau=0.005 (0.5%). Dette betyr at target-criticen reagerer 4x tregere pa endringer i live-criticen, og fungerer som en stabilisator ved brå reward-endringer.

I den stabile fasen (280k-450k) var criticen allerede veldig nøyaktig med lave gradienter. En tregere target ville ikke bremset laeringen her. Men ved kollapsen (460k-500k) ville den dempet den positive feedback-loopen mellom live critic og target critic.

### Endring 2: Strammere critic gradient clipping (critic_grad_clip: 10 → 2.0)

**Begrunnelse:** I den stabile fasen var critic_grad_norm mellom 0.3 og 0.65. Under kollapsen steg den til 2.0. Gjeldende clip pa 10 var aldri aktiv — den ga ingen beskyttelse. Ved a sette clip til 2.0 får vi:
- Ingen effekt i stabil fase (gradienter er 0.3-0.65, godt under 2.0)
- Aktiv begrensning under kollaps (gradienter kuttes fra potensielt >2.0)

### Endring 3: Halvert critic learning rate (critic_lr: 1e-4 → 5e-5)

**Begrunnelse:** Kombinert med tregere target-oppdateringer gir lavere LR en generelt mer konservativ critic. Den stabile laeringsfasen viste at agenten kan laere Pong med lave critic-gradienter — vi trenger ikke rask critic-tilpasning.

### Forventet effekt

Disse endringene bremser critic-tilpasningen ved regime-overganger. Risikoen er at criticen bruker lenger tid på a tilpasse seg i tidlig trening (fase 1 kan bli lengre). Men gitt at v6 brukte ~80k steps pa oppvarming uansett, og den stabile fasen varte ~170k steps, er dette en akseptabel trade-off.

Mal: Unnga kollapsen etter 460k og la den stabile laeringstrajektorien (-17 til +11) fortsette forbi +11 mot +19-nivå.

### Alle v7-parameterendringer

| Parameter        | v6      | v7      | Endring         |
|------------------|---------|---------|-----------------|
| critic_ema_decay | 0.98    | 0.995   | 4x tregere target |
| critic_grad_clip | 10.0    | 2.0     | 5x strammere    |
| critic_lr        | 1.0e-4  | 5.0e-5  | 2x lavere       |

Alt annet er identisk med v6.

---

## v7-resultater (Wandb ID: `cmfk0kzg`, 500k steps, ~35 timer)

### Treningsforlop

v7 startet fra scratch (ingen resume) med de tre critic-endringene. Treningskurven var markant forskjellig fra v6:

```
Step    Reward   Value    Critic Loss  Critic Grad
124k    -16      +1.50    0.304        1.65
192k    -19      -1.31    0.087        0.80
268k    -10      -1.16    0.092        0.85      ← peak #1
320k    -18      -0.76    0.122        0.95
340k    -12      -0.43    0.145        1.07
360k    -15      +0.03    0.171        1.19
400k    -18      +0.62    0.210        1.43
456k     -8      -0.91    0.338        1.99      ← peak #2, grad clip aktiveres
484k    -17      -1.19    0.365        2.22      ← clipped, faller tilbake
501k    -19      -1.15    0.379        2.31      ← slutt
```

### Observasjoner

**Ingen eksplosiv kollaps.** I motsetning til v6, kollapset aldri criticen bratt. Value-estimater og gradienter var jevne gjennom hele runnen. Critic-stabiliseringen fungerte som intendert.

**Men den laerte aldri.** Peak reward var -8 (ved 456k), og den falt tilbake til -19 ved slutt. Reward-kurven var preget av konstant oscillering mellom -10 og -20, uten den glatte monotone stigningen som kjennetegnet v6.

**Grad clip aktiverte seg og blokkerte korreksjoner.** Ved ~460k passerte critic_grad_norm clippen pa 2.0. Etter det kunne criticen ikke gjøre store nok oppdateringer til a følge reward-endringer, og den ble effektivt laast.

### Hvorfor v7 underpresterte: Bootstrapping-problemet

v6 (som ble resumert) hadde en **moden world model** nar AC-treningen startet effektivt. Criticen fikk meningsfulle rollouts fra dag 1 og konvergerte raskt.

v7 startet AC etter bare 20k steps (ac_warmup). Pa det tidspunktet var wm/pred fortsatt ~0.25 (vs. 0.013 nar v6 ble resumert). Criticen trente pa **søppel-rollouts** i ~150k steps og laerte feil verdi-estimater som den deretter matte avlaere.

Resume-effekten i v6 fungerte som en utilsiktet **metric gate**: AC-treningen startet først nar world model var moden nok til a produsere meningsfulle rollouts.

### v7 vs v6 — sammenligning

| Metrikk              | v6 (resumed)         | v7 (fra scratch)     |
|----------------------|----------------------|----------------------|
| Peak reward          | **+11**              | -8                   |
| Slutt-reward         | -12                  | -19                  |
| Kollaps?             | Ja (460k, eksplosiv) | Nei (stabil men flat) |
| Critic grad maks     | 2.0                  | 2.3 (clipped)        |
| WM pred ved AC-start | 0.013                | ~0.25                |
| Laeringsmønster      | Jevn, monoton        | Oscillerende         |

**Konklusjon:** Critic-stabilisering alene er utilstrekkelig. Bootstrapping-problemet (AC pa umodne rollouts) er den dominerende flaskehalsen.

---

## v8-plan: Metric-gated AC-start

### Hypotese

v6s resume-eksperiment viste utilsiktet at sen AC-oppstart med moden world model gir jevnere og raskere laering. v8 formaliserer dette med en eksplisitt metric gate.

### Implementasjon

Ny config-parameter `ac_gate_threshold` i `train_jepa.py`:
- World model trener alene fra start (collection med random/learned policy)
- En EMA (decay=0.99) av `wm/pred` trackes
- AC-trening starter først nar EMA < `ac_gate_threshold` OG `global_step >= ac_warmup_steps`
- Gate-passeringen logges med nøyaktig step og pred-verdi

### Valg av threshold

Fra v7-dataen:
- `wm/pred = 0.016` ved ~160k steps
- `wm/pred = 0.013` ved ~300k steps
- v6 ble resumert med pred ~0.013

Threshold settes til **0.015**. Dette tilsvarer omtrent 120-150k steps world model-only trening — nok til at predictoren er konvergert, men ikke sa sent at vi kaster bort compute.

### Endringer fra v7

| Parameter          | v7       | v8       | Endring                |
|--------------------|----------|----------|------------------------|
| ac_gate_threshold  | (ingen)  | 0.015    | Metric-gated AC start  |
| critic_ema_decay   | 0.995    | 0.995    | Uendret (fra v7)       |
| critic_grad_clip   | 2.0      | 2.0      | Uendret (fra v7)       |
| critic_lr          | 5.0e-5   | 5.0e-5   | Uendret (fra v7)       |

### Forventet resultat

Criticen starter med en moden world model (pred < 0.015), noe som eliminerer søppel-bootstrapping-fasen. Forventet:
- Raskere og jevnere reward-forbedring (som v6 fase 2)
- Critic-stabiliseringen fra v7 forhindrer kollaps nar rewards gar positive
- Malet er reward > +11 uten kollaps

### Fremtidige versjoner

- **v9**: Residual prediction (z_next = z_prev + Predictor(z_prev, a)). World model-endring, ortogonal til AC-endringene i v7/v8.
- **v10+**: Vurdere lengre rollouts (2→4 step), evt. fjerne aux decoder.

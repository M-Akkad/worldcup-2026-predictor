/* i18n.js, tiny translation layer (NL default, EN). No build step.
   Static text uses data-i18n / data-i18n-html / data-i18n-ph / data-i18n-title /
   data-i18n-aria attributes; dynamic strings call t(key, vars). */
"use strict";

const I18N = {
  nl: {
    meta_desc: "Live titelkansen voor het WK 2026, Elo, Dixon-Coles en Monte Carlo, geconditioneerd op echte uitslagen.",
    nav_aria: "Hoofdnavigatie",
    nav_dashboard: "Dashboard", nav_groups: "Groepen", nav_results: "Resultaten",
    nav_predict: "Voorspeller", nav_data: "Data & model",
    lang_title: "Switch to English",

    // model option labels
    opt_ensemble: "Ensemble (aanbevolen)", opt_elo: "Elo (sterk op doelpunten)",
    opt_olr: "Ordered logit (wie wint)",

    // chips
    chip_matches: "<strong>{n}</strong> wedstrijden",
    chip_cond: "<strong>{n}</strong> WK-uitslagen verwerkt",
    chip_waiting: "wacht op de aftrap",
    chip_run: "simulatie: <strong>{model}</strong> · {n}×",
    chip_norun: "nog geen simulatie",

    // dashboard
    hero_loading: "Laden …",
    eyebrow_top10: "Titelkansen · top 10",
    h_titlerace: "Titelrace door de tijd",
    p_titlerace: "Elke simulatie-run is één meetpunt, draai er dagelijks één na het ophalen van de uitslagen en zie de kansen verschuiven.",
    h_all48: "Alle 48 landen",
    p_all48: "Klik op een kolomkop om te sorteren. Beweeg over een kop voor uitleg.",
    explain_read: "Hoe lees ik dit?",
    explain_elo: "<b>Elo</b>, de speelsterkte van een land, opgebouwd uit alle interlands sinds 1872. Hoger is sterker; het gemiddelde ligt rond 1500, de top zit boven 2000. Elke gespeelde wedstrijd past het cijfer aan.",
    explain_groupwin: "<b>Groepswinst</b>, kans dat het land zijn groep van vier als nummer 1 afsluit.",
    explain_ko: "<b>Knock-out → Titel</b>, kans om elke volgende ronde te halen: de laatste 32, kwartfinale, halve finale, finale, en uiteindelijk wereldkampioen. Deze kansen lopen per ronde af.",
    explain_mc: "Alle percentages komen uit een <b>Monte-Carlo-simulatie</b>: het toernooi wordt tienduizenden keren nagespeeld en we tellen hoe vaak elk land hoever komt. Al gespeelde WK-wedstrijden worden vastgezet, dus de kansen volgen de echte uitslagen.",
    search_ph: "Zoek een land…", search_aria: "Zoek een land",
    hero_norun: 'Nog geen simulatie gedraaid. Ga naar <a href="#" onclick="showTab(\'data\');return false;">Data &amp; model</a> en start er één, de uitslag staat hier binnen een halve minuut.',
    hero_fav_eyebrow: "favoriet volgens {model} · {n} simulaties",
    fav_worldtitle: "wereldtitel", fav_final: "finale",
    cond_main: "Geconditioneerd op {g} gespeelde groepswedstrijden",
    cond_ko: " en {n} besliste knock-outs",

    // table columns + tips
    col_team: "Team", col_group: "Groep", col_elo: "Elo", col_groupwin: "Groepswinst",
    col_ko: "Knock-out", col_qf: "Kwartfinale", col_sf: "Halve finale",
    col_final: "Finale", col_title: "Titel",
    tip_team: "Land", tip_group: "Groep (A–L)",
    tip_elo: "Speelsterkte uit alle interlands; ~1500 gemiddeld, top boven 2000",
    tip_groupwin: "Kans om de groep als nummer 1 af te sluiten",
    tip_ko: "Kans om de groepsfase te overleven (laatste 32)",
    tip_qf: "Kans om de kwartfinale te bereiken (laatste 8)",
    tip_sf: "Kans om de halve finale te bereiken (laatste 4)",
    tip_final: "Kans om de finale te bereiken",
    tip_title: "Kans om wereldkampioen te worden",
    p_history_grow: "De titelkans-grafiek groeit hier zodra er meerdere simulaties zijn gedraaid (bijv. dagelijks tijdens het toernooi).",
    chart_aria: "Titelkansen per simulatie",

    // groups
    h_groups: "De twaalf groepen",
    groups_note: "Kansen verschijnen zodra de eerste simulatie is gedraaid; gespeelde uitslagen komen automatisch onder de groep te staan.",
    group_label: "Groep {g}", badge_host: "HOST",

    // results
    h_fixtures: "Aankomende wedstrijden, wie tegen wie, en wat het model verwacht",
    p_fixtures: "De eerstvolgende WK-wedstrijden uit het speelschema met de voorspelling van het model: de winkans per uitkomst (thuiswinst / gelijkspel / uitwinst), het verwachte aantal goals en de meest waarschijnlijke eindstand. THUIS = gastland speelt thuis. Een exacte eindstand heeft altijd een lage kans, voetbal is grillig; kijk vooral naar de winkansen.",
    fixtures_empty: "Geen aankomende WK-wedstrijden in de data.",
    h_scorecard: "Voorspeld vs. werkelijk",
    p_scorecard: "Voor elke al gespeelde WK-wedstrijd: wat het model vóór de aftrap voorspelde (lekvrij, met de Elo van net vóór de wedstrijd) naast de echte uitslag, en het verschil. Kans op echte uitslag is de kans die het model gaf aan wat er écht gebeurde, hoe hoger, hoe beter het model het zag aankomen.",
    scorecard_empty: "Nog geen gespeelde WK-wedstrijden in de data. Haal uitslagen op via Data & model → Haal uitslagen op zodra het toernooi begint.",
    fx_calc: "Voorspellingen berekenen…",
    th_date: "Datum", th_match: "Wedstrijd",
    th_winprob: "Winkans", sub_hda: "(thuis / gelijk / uit)",
    th_expgoals: "Verw. goals", tip_expgoals: "Verwachte goals: het gemiddelde aantal doelpunten dat het model voor elk team voorspelt",
    th_expresult: "Verwachte uitslag", th_favourite: "Favoriet", badge_home: "THUIS",
    sc_calc: "Berekenen…",
    sc_n: "gespeelde WK-wedstrijden", sc_fav: "favoriet kwam uit",
    sc_pact: "gem. kans op de echte uitslag",
    sc_quality: "gemiddelde kwaliteitsscore", sc_lower_better: "(lager = beter)",
    tip_rps_quality: "Ranked Probability Score: meet hoe goed de voorspelde kansen waren. 0 = perfect, lager is beter",
    th_predicted: "Voorspeld", th_result: "Uitslag",
    th_realchance: "Kans op echte uitslag",
    tip_realchance: "De kans die het model vóór de wedstrijd gaf aan wat er écht gebeurde, hoger is beter",

    // predictor
    h_predict: "Voorspel één wedstrijd",
    lbl_home1: "Thuis / team 1", lbl_away1: "Uit / team 2", lbl_model: "Model",
    lbl_neutral: "Neutraal terrein",
    lbl_odds: "Bookmaker-odds (thuis, gelijk, uit, optioneel)",
    lbl_mw: "Marktgewicht (0–1)",
    title_mw: "Hoe zwaar de bookmaker-odds meewegen: 0 = alleen het model, 1 = alleen de markt",
    btn_predict: "Voorspel",
    p_shin: "Odds worden eerst ontdaan van de bookmakermarge via Shin's methode (corrigeert de favourite-longshot-bias) en daarna gemengd met het model.",
    eyebrow_goals: "Doelpunten", eyebrow_scorelines: "Meest waarschijnlijke uitslagen",
    eyebrow_scoreprobs: "Scorekansen (thuis ↓ / uit →)",
    venue_neutral: "neutraal terrein", venue_home: "thuiswedstrijd",
    pred_win: "winst {team}", pred_draw: "gelijkspel",
    pred_market: "Markt (Shin de-vig): {probs}, gemengd in de cijfers hierboven.",
    goal_total: "verwachte goals totaal", goal_over25: "over 2.5 goals",
    goal_under25: "under 2.5: {x}", goal_btts: "beide teams scoren",
    goal_over1535: "over 1.5: {a} · over 3.5: {b}",
    goal_hint: 'Doelpunten-kansen zijn het best gekalibreerd met het <b>elo</b>-model (zie validatie); {model} is iets scherper op wie-wint maar overschat het aantal goals licht.',

    // data & model
    h_fetch: "Uitslagen binnenhalen",
    p_fetch: "De gratis openfootball-feed (geen API-key, ~dagelijks bijgewerkt) of handmatig per wedstrijd. Dubbele invoer wordt automatisch overgeslagen.",
    btn_fetch: "Haal uitslagen op (gratis feed)",
    lbl_home: "Thuis", lbl_away: "Uit", lbl_hs: "T", lbl_as: "U", lbl_date: "Datum",
    lbl_homegame: "Thuiswedstrijd gastland", btn_add: "Voeg toe",
    h_news: "Nieuws-correcties",
    p_news: "Blessures, schorsingen, vorm, als begrensde Elo-delta. Wordt toegepast op de eerstvolgende simulatie.",
    lbl_team: "Team", lbl_delta: "Delta", lbl_reason: "Reden",
    ph_reason: "spits geblesseerd", btn_save: "Opslaan",
    h_sim: "Simulatie draaien", lbl_nsims: "Aantal simulaties",
    lbl_applyadj: "Pas correcties toe", btn_sim: "Start simulatie",
    h_backtest: "Backtest",
    p_backtest: "Eerlijke meting op wedstrijden die het model niet kende: lager dan de base-rate-kolom (simpelweg de gemiddelde uitslag-verdeling gokken) betekent echte voorspelkracht.",
    bt_explain: "Wat betekenen deze cijfers?",
    bt_li_rps: "<b>RPS</b>, <i>Ranked Probability Score</i>: de hoofdmaat voor de kwaliteit van de kansen. 0 = perfect, lager is beter.",
    bt_li_ll: "<b>log-loss</b>, straft zelfverzekerde missers zwaar. Lager is beter.",
    bt_li_brier: "<b>Brier</b>, gemiddelde kwadratische fout van de kansen. Lager is beter.",
    bt_li_acc: "<b>accuracy</b>, hoe vaak de meest waarschijnlijke uitkomst klopte (de zwakste maat, negeert hoe zeker het model was).",
    bt_li_ece: "<b>ECE</b>, <i>kalibratiefout</i>: zegt het model 30% en gebeurt het ook ~30% van de tijd? Dichter bij 0 = betrouwbaarder.",
    lbl_from: "Vanaf", btn_measure: "Meet",
    h_tune: "Parameters optimaliseren",
    p_tune: "Zoekt automatisch de beste modelinstellingen, Elo K-factoren, thuisvoordeel, de goals-brug en het ensemble-gewicht, door slim (Bayesiaans) te zoeken en elke combinatie eerlijk out-of-sample te toetsen. Lagere score = beter. Een gevonden config wordt pas aangeraden als hij de huidige standaard verslaat.",
    lbl_metric: "Maatstaf", lbl_trials: "Aantal pogingen", btn_tune: "Start optimalisatie",

    // dynamic data & model
    adj_empty: "Geen actieve correcties. Voeg er één toe op basis van blessurenieuws, begrensd tot ±100 Elo, zodat context kan bijsturen maar nooit domineren.",
    adj_delete: "Verwijder", toast_adj_saved: "Correctie opgeslagen",
    toast_fetch: "{n} uitslag(en) bijgewerkt",
    toast_fetch_none: "Geen nieuwe uitslagen, de gratis feed (≈dagelijks) loopt mogelijk nog achter. Voer de uitslag hieronder handmatig in.",
    toast_result_added: "Uitslag toegevoegd",
    toast_result_updated: "Uitslag bijgewerkt (gecorrigeerd)",
    toast_result_dupe: "Stond er al in (overgeslagen)",
    sim_started: "Simulatie gestart …", sim_busy: "Bezig … {s}s",
    sim_done: "Klaar in {s}s (run #{id}).", toast_sim_done: "Simulatie klaar",
    job_failed: "Mislukt: {e}",
    bt_running: "Backtest draait …",
    th_metric: "maatstaf", th_model: "model", th_baserate: "base-rate",
    tip_baserate: "Simpelweg de gemiddelde uitslag-verdeling gokken, de lat die het model moet verslaan",
    bt_ece_line: "{x} · n={n} wedstrijden · lager dan base-rate = echte voorspelkracht.",
    tip_m_rps: "Ranked Probability Score, hoofdmaat voor de kwaliteit van de kansen; lager is beter",
    tip_m_ll: "Straft zelfverzekerde missers zwaar; lager is beter",
    tip_m_brier: "Gemiddelde kwadratische fout van de kansen; lager is beter",
    tip_m_acc: "Hoe vaak de meest waarschijnlijke uitkomst klopte; hoger is beter",
    tip_m_ece: "Kalibratiefout: zegt het model 30% en gebeurt het ook ~30%? Dichter bij 0 = beter",
    tune_started: "Optimalisatie gestart …", tune_trial: "Poging {done}/{total}",
    tune_best: " · beste {ml} {v}", tune_done: "Klaar in {s}s.",
    toast_tune_better: "Betere parameters gevonden", toast_tune_same: "Standaard blijft de beste",
    verdict_better: "beter dan de standaard", verdict_worse: "niet beter, houd de standaard",
    tune_vs: "vs standaard {v}", tune_meta: "· {n} pogingen · model {m}",
    th_found: "gevonden", th_default: "standaard",
    tune_bestparams: "Beste parameters", tune_adopt: " (overnemen als standaard?)",
    tune_note_better: "Deze instellingen verslaan de huidige standaard out-of-sample. Laat ze vastzetten in de modeldefaults na een groene testronde.",
    tune_note_worse: "Geen verbetering gevonden. Probeer meer pogingen of een ander model.",
  },

  en: {
    meta_desc: "Live World Cup 2026 title odds, Elo, Dixon-Coles and Monte Carlo, conditioned on real results.",
    nav_aria: "Main navigation",
    nav_dashboard: "Dashboard", nav_groups: "Groups", nav_results: "Results",
    nav_predict: "Predictor", nav_data: "Data & model",
    lang_title: "Schakel naar Nederlands",

    opt_ensemble: "Ensemble (recommended)", opt_elo: "Elo (strong on goals)",
    opt_olr: "Ordered logit (who wins)",

    chip_matches: "<strong>{n}</strong> matches",
    chip_cond: "<strong>{n}</strong> WC results in",
    chip_waiting: "awaiting kickoff",
    chip_run: "simulation: <strong>{model}</strong> · {n}×",
    chip_norun: "no simulation yet",

    hero_loading: "Loading …",
    eyebrow_top10: "Title odds · top 10",
    h_titlerace: "Title race over time",
    p_titlerace: "Each simulation run is one data point, run one daily after fetching results and watch the odds shift.",
    h_all48: "All 48 teams",
    p_all48: "Click a column header to sort. Hover a header for an explanation.",
    explain_read: "How do I read this?",
    explain_elo: "<b>Elo</b>, a team's playing strength, built from every international since 1872. Higher is stronger; the average sits around 1500, the top above 2000. Each match nudges the number.",
    explain_groupwin: "<b>Group win</b>, the chance the team finishes its group of four in first place.",
    explain_ko: "<b>Knockout → Title</b>, the chance of reaching each next round: the last 32, quarter-final, semi-final, final, and ultimately world champion. These chances fall off round by round.",
    explain_mc: "Every percentage comes from a <b>Monte Carlo simulation</b>: the tournament is replayed tens of thousands of times and we count how far each team gets. Already-played WC matches are pinned, so the odds follow the real results.",
    search_ph: "Search a team…", search_aria: "Search a team",
    hero_norun: 'No simulation run yet. Go to <a href="#" onclick="showTab(\'data\');return false;">Data &amp; model</a> and start one, the result shows up here within half a minute.',
    hero_fav_eyebrow: "favourite per {model} · {n} simulations",
    fav_worldtitle: "world title", fav_final: "final",
    cond_main: "Conditioned on {g} played group matches",
    cond_ko: " and {n} decided knockouts",

    col_team: "Team", col_group: "Group", col_elo: "Elo", col_groupwin: "Group win",
    col_ko: "Knockout", col_qf: "Quarter-final", col_sf: "Semi-final",
    col_final: "Final", col_title: "Title",
    tip_team: "Team", tip_group: "Group (A–L)",
    tip_elo: "Playing strength from all internationals; ~1500 average, top above 2000",
    tip_groupwin: "Chance of finishing the group in first place",
    tip_ko: "Chance of surviving the group stage (last 32)",
    tip_qf: "Chance of reaching the quarter-final (last 8)",
    tip_sf: "Chance of reaching the semi-final (last 4)",
    tip_final: "Chance of reaching the final",
    tip_title: "Chance of becoming world champion",
    p_history_grow: "The title-odds chart grows here once several simulations have been run (e.g. daily during the tournament).",
    chart_aria: "Title odds per simulation",

    h_groups: "The twelve groups",
    groups_note: "Odds appear once the first simulation has run; played results are shown automatically under each group.",
    group_label: "Group {g}", badge_host: "HOST",

    h_fixtures: "Upcoming matches, who plays whom, and what the model expects",
    p_fixtures: "The next WC matches from the schedule with the model's prediction: the win chance per outcome (home win / draw / away win), the expected number of goals and the most likely final score. HOME = host playing at home. An exact final score is always low-probability, football is noisy; focus on the win chances.",
    fixtures_empty: "No upcoming WC matches in the data.",
    h_scorecard: "Predicted vs. actual",
    p_scorecard: "For every played WC match: what the model predicted before kickoff (leakage-free, with the Elo from just before the match) next to the real result, and the gap. Chance of real result is the probability the model gave to what actually happened, the higher, the better it saw it coming.",
    scorecard_empty: "No played WC matches in the data yet. Fetch results via Data & model → Fetch results once the tournament starts.",
    fx_calc: "Computing predictions…",
    th_date: "Date", th_match: "Match",
    th_winprob: "Win chance", sub_hda: "(home / draw / away)",
    th_expgoals: "Exp. goals", tip_expgoals: "Expected goals: the average number of goals the model predicts for each team",
    th_expresult: "Predicted score", th_favourite: "Favourite", badge_home: "HOME",
    sc_calc: "Computing…",
    sc_n: "WC matches played", sc_fav: "favourite came through",
    sc_pact: "avg. chance of the real result",
    sc_quality: "average quality score", sc_lower_better: "(lower = better)",
    tip_rps_quality: "Ranked Probability Score: measures how good the predicted probabilities were. 0 = perfect, lower is better",
    th_predicted: "Predicted", th_result: "Result",
    th_realchance: "Chance of real result",
    tip_realchance: "The probability the model gave before the match to what actually happened, higher is better",

    h_predict: "Predict one match",
    lbl_home1: "Home / team 1", lbl_away1: "Away / team 2", lbl_model: "Model",
    lbl_neutral: "Neutral venue",
    lbl_odds: "Bookmaker odds (home, draw, away, optional)",
    lbl_mw: "Market weight (0–1)",
    title_mw: "How much the bookmaker odds count: 0 = model only, 1 = market only",
    btn_predict: "Predict",
    p_shin: "Odds are first stripped of the bookmaker margin via Shin's method (correcting the favourite-longshot bias) and then blended with the model.",
    eyebrow_goals: "Goals", eyebrow_scorelines: "Most likely scorelines",
    eyebrow_scoreprobs: "Scoreline odds (home ↓ / away →)",
    venue_neutral: "neutral venue", venue_home: "home match",
    pred_win: "{team} win", pred_draw: "draw",
    pred_market: "Market (Shin de-vig): {probs}, blended into the figures above.",
    goal_total: "expected goals total", goal_over25: "over 2.5 goals",
    goal_under25: "under 2.5: {x}", goal_btts: "both teams to score",
    goal_over1535: "over 1.5: {a} · over 3.5: {b}",
    goal_hint: 'Goal odds are best calibrated with the <b>elo</b> model (see validation); {model} is a touch sharper on who-wins but slightly over-predicts goals.',

    h_fetch: "Fetch results",
    p_fetch: "The free openfootball feed (no API key, updated ~daily) or by hand per match. Duplicate entries are skipped automatically.",
    btn_fetch: "Fetch results (free feed)",
    lbl_home: "Home", lbl_away: "Away", lbl_hs: "H", lbl_as: "A", lbl_date: "Date",
    lbl_homegame: "Host playing at home", btn_add: "Add",
    h_news: "News adjustments",
    p_news: "Injuries, suspensions, form, as a bounded Elo delta. Applied to the next simulation.",
    lbl_team: "Team", lbl_delta: "Delta", lbl_reason: "Reason",
    ph_reason: "striker injured", btn_save: "Save",
    h_sim: "Run simulation", lbl_nsims: "Number of simulations",
    lbl_applyadj: "Apply adjustments", btn_sim: "Start simulation",
    h_backtest: "Backtest",
    p_backtest: "An honest measurement on matches the model didn't know: lower than the base-rate column (simply guessing the average outcome distribution) means real predictive skill.",
    bt_explain: "What do these numbers mean?",
    bt_li_rps: "<b>RPS</b>, <i>Ranked Probability Score</i>: the headline measure of probability quality. 0 = perfect, lower is better.",
    bt_li_ll: "<b>log-loss</b>, punishes confident misses heavily. Lower is better.",
    bt_li_brier: "<b>Brier</b>, mean squared error of the probabilities. Lower is better.",
    bt_li_acc: "<b>accuracy</b>, how often the most likely outcome was right (the weakest measure, ignores how confident the model was).",
    bt_li_ece: "<b>ECE</b>, <i>calibration error</i>: when the model says 30%, does it happen ~30% of the time? Closer to 0 = more trustworthy.",
    lbl_from: "From", btn_measure: "Measure",
    h_tune: "Optimise parameters",
    p_tune: "Automatically searches for the best model settings, Elo K-factors, home advantage, the goals bridge and the ensemble weight, by searching smartly (Bayesian) and testing each combination fairly out-of-sample. Lower score = better. A found config is only recommended once it beats the current default.",
    lbl_metric: "Metric", lbl_trials: "Number of trials", btn_tune: "Start optimisation",

    adj_empty: "No active adjustments. Add one based on injury news, capped at ±100 Elo, so context can steer but never dominate.",
    adj_delete: "Remove", toast_adj_saved: "Adjustment saved",
    toast_fetch: "{n} result(s) updated",
    toast_fetch_none: "No new results, the free feed (~daily) may be lagging. Enter the result by hand below.",
    toast_result_added: "Result added",
    toast_result_updated: "Result updated (corrected)",
    toast_result_dupe: "Already present (skipped)",
    sim_started: "Simulation started …", sim_busy: "Working … {s}s",
    sim_done: "Done in {s}s (run #{id}).", toast_sim_done: "Simulation done",
    job_failed: "Failed: {e}",
    bt_running: "Backtest running …",
    th_metric: "metric", th_model: "model", th_baserate: "base-rate",
    tip_baserate: "Simply guessing the average outcome distribution, the bar the model must beat",
    bt_ece_line: "{x} · n={n} matches · lower than base-rate = real predictive skill.",
    tip_m_rps: "Ranked Probability Score, headline measure of probability quality; lower is better",
    tip_m_ll: "Punishes confident misses heavily; lower is better",
    tip_m_brier: "Mean squared error of the probabilities; lower is better",
    tip_m_acc: "How often the most likely outcome was right; higher is better",
    tip_m_ece: "Calibration error: when the model says 30%, does it happen ~30%? Closer to 0 = better",
    tune_started: "Optimisation started …", tune_trial: "Trial {done}/{total}",
    tune_best: " · best {ml} {v}", tune_done: "Done in {s}s.",
    toast_tune_better: "Better parameters found", toast_tune_same: "Default stays best",
    verdict_better: "better than the default", verdict_worse: "not better, keep the default",
    tune_vs: "vs default {v}", tune_meta: "· {n} trials · model {m}",
    th_found: "found", th_default: "default",
    tune_bestparams: "Best parameters", tune_adopt: " (adopt as default?)",
    tune_note_better: "These settings beat the current default out-of-sample. Have them locked into the model defaults after a green test run.",
    tune_note_worse: "No improvement found. Try more trials or a different model.",
  },
};

let LANG = localStorage.getItem("lang") || "nl";

function t(key, vars) {
  let s = (I18N[LANG] && I18N[LANG][key]);
  if (s == null) s = (I18N.nl[key] != null ? I18N.nl[key] : key);
  if (vars) for (const k in vars) s = s.split("{" + k + "}").join(vars[k]);
  return s;
}

function applyI18n(root) {
  root = root || document;
  root.querySelectorAll("[data-i18n]").forEach(el => {
    const val = t(el.dataset.i18n);
    // a label like `<label>Text<input></label>` must keep its nested control:
    // only replace the leading text node, never the whole subtree.
    if (el.children.length && el.firstChild && el.firstChild.nodeType === 3) {
      el.firstChild.nodeValue = val;
    } else {
      el.textContent = val;
    }
  });
  root.querySelectorAll("[data-i18n-html]").forEach(el => { el.innerHTML = t(el.dataset.i18nHtml); });
  root.querySelectorAll("[data-i18n-ph]").forEach(el => el.setAttribute("placeholder", t(el.dataset.i18nPh)));
  root.querySelectorAll("[data-i18n-title]").forEach(el => el.setAttribute("title", t(el.dataset.i18nTitle)));
  root.querySelectorAll("[data-i18n-aria]").forEach(el => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
  const desc = document.querySelector('meta[name="description"]');
  if (desc) desc.setAttribute("content", t("meta_desc"));
  document.documentElement.lang = LANG;
  const lb = document.getElementById("lang-btn");
  if (lb) { lb.textContent = LANG === "nl" ? "EN" : "NL"; lb.title = t("lang_title"); }
}

function setLang(l) {
  if (l !== "nl" && l !== "en") return;
  LANG = l;
  localStorage.setItem("lang", l);
  applyI18n();
  window.dispatchEvent(new CustomEvent("langchange"));
}

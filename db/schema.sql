CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    date        TEXT NOT NULL,
    location    TEXT,
    url         TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS fighters (
    fighter_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    nickname        TEXT,
    dob             TEXT,
    height_inches   REAL,
    reach_inches    REAL,
    stance          TEXT,
    weight_class    TEXT,
    url             TEXT UNIQUE,
    sherdog_url     TEXT,
    total_wins      INTEGER DEFAULT 0,
    total_losses    INTEGER DEFAULT 0,
    total_draws     INTEGER DEFAULT 0,
    ko_wins         INTEGER DEFAULT 0,
    sub_wins        INTEGER DEFAULT 0,
    dec_wins        INTEGER DEFAULT 0,
    ko_losses       INTEGER DEFAULT 0,
    sub_losses      INTEGER DEFAULT 0,
    dec_losses      INTEGER DEFAULT 0,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS fights (
    fight_id        TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL REFERENCES events(event_id),
    fighter_a_id    TEXT NOT NULL REFERENCES fighters(fighter_id),
    fighter_b_id    TEXT NOT NULL REFERENCES fighters(fighter_id),
    winner_id       TEXT REFERENCES fighters(fighter_id),
    method          TEXT,
    method_detail   TEXT,
    round           INTEGER,
    time            TEXT,
    time_format     TEXT,
    is_title_fight  INTEGER DEFAULT 0,
    weight_class    TEXT,
    url             TEXT UNIQUE,
    fight_date      TEXT
);

CREATE TABLE IF NOT EXISTS fight_stats (
    stat_id             TEXT PRIMARY KEY,
    fight_id            TEXT NOT NULL REFERENCES fights(fight_id),
    fighter_id          TEXT NOT NULL REFERENCES fighters(fighter_id),
    round               INTEGER NOT NULL,
    knockdowns          INTEGER DEFAULT 0,
    sig_str_landed      INTEGER DEFAULT 0,
    sig_str_attempted   INTEGER DEFAULT 0,
    sig_str_pct         REAL,
    total_str_landed    INTEGER DEFAULT 0,
    total_str_attempted INTEGER DEFAULT 0,
    td_landed           INTEGER DEFAULT 0,
    td_attempted        INTEGER DEFAULT 0,
    td_pct              REAL,
    sub_attempts        INTEGER DEFAULT 0,
    rev                 INTEGER DEFAULT 0,
    ctrl_seconds        INTEGER DEFAULT 0,
    head_landed         INTEGER DEFAULT 0,
    head_attempted      INTEGER DEFAULT 0,
    body_landed         INTEGER DEFAULT 0,
    body_attempted      INTEGER DEFAULT 0,
    leg_landed          INTEGER DEFAULT 0,
    leg_attempted       INTEGER DEFAULT 0,
    distance_landed     INTEGER DEFAULT 0,
    distance_attempted  INTEGER DEFAULT 0,
    clinch_landed       INTEGER DEFAULT 0,
    clinch_attempted    INTEGER DEFAULT 0,
    ground_landed       INTEGER DEFAULT 0,
    ground_attempted    INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS odds_history (
    odds_id         TEXT PRIMARY KEY,
    fight_id        TEXT REFERENCES fights(fight_id),
    fighter_a_id    TEXT REFERENCES fighters(fighter_id),
    fighter_b_id    TEXT REFERENCES fighters(fighter_id),
    fight_date      TEXT,
    fighter_a_name  TEXT,
    fighter_b_name  TEXT,
    fighter_a_odds  INTEGER,
    fighter_b_odds  INTEGER,
    book            TEXT DEFAULT 'betmma',
    scraped_at      TEXT
);

CREATE TABLE IF NOT EXISTS elo_history (
    elo_id          TEXT PRIMARY KEY,
    fighter_id      TEXT NOT NULL REFERENCES fighters(fighter_id),
    fight_id        TEXT NOT NULL REFERENCES fights(fight_id),
    fight_date      TEXT NOT NULL,
    elo_before      REAL NOT NULL,
    elo_after       REAL NOT NULL,
    opponent_id     TEXT REFERENCES fighters(fighter_id),
    result          TEXT
);

CREATE INDEX IF NOT EXISTS idx_fights_date ON fights(fight_date);
CREATE INDEX IF NOT EXISTS idx_fights_event ON fights(event_id);
CREATE INDEX IF NOT EXISTS idx_fight_stats_fight ON fight_stats(fight_id);
CREATE INDEX IF NOT EXISTS idx_fight_stats_fighter ON fight_stats(fighter_id);
CREATE INDEX IF NOT EXISTS idx_elo_fighter ON elo_history(fighter_id);
CREATE INDEX IF NOT EXISTS idx_elo_date ON elo_history(fight_date);
CREATE INDEX IF NOT EXISTS idx_odds_fight ON odds_history(fight_id);

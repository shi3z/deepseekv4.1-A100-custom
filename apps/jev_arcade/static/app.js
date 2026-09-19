// Jev Arcade 20 - Client Application Logic

(function() {
    'use strict';

    // App State
    let games = [];
    let currentGameIndex = 0;
    let currentEngine = 'local'; // 'local' | 'official'
    let currentFilter = 'all';

    // Stats in localStorage
    const savedStats = JSON.parse(localStorage.getItem('jev_arcade_stats') || '{"plays": 0, "totalMs": 0}');
    let stats = savedStats;

    // DOM Elements
    const gamesGrid = document.getElementById('games-grid');
    const filterTabs = document.querySelectorAll('.filter-tab');
    const engineLocalBtn = document.getElementById('engine-local-btn');
    const engineCloudBtn = document.getElementById('engine-cloud-btn');
    const soundBtn = document.getElementById('sound-btn');
    const soundIcon = document.getElementById('sound-icon');
    const playCounter = document.getElementById('play-counter');
    const avgLatency = document.getElementById('avg-latency');
    const lastEngineLabel = document.getElementById('last-engine-label');

    // Modal Elements
    const modal = document.getElementById('game-modal');
    const modalCloseBtn = document.getElementById('modal-close-btn');
    const modalEmoji = document.getElementById('modal-emoji');
    const modalTag = document.getElementById('modal-tag');
    const modalGameId = document.getElementById('modal-game-id');
    const modalTitle = document.getElementById('modal-title');
    const modalDesc = document.getElementById('modal-desc');
    const modalScenario = document.getElementById('modal-scenario');
    const modalPresets = document.getElementById('modal-presets');
    const modalInputLabel = document.getElementById('modal-input-label');
    const userTextInput = document.getElementById('user-text-input');
    const playSubmitBtn = document.getElementById('play-submit-btn');
    const clearInputBtn = document.getElementById('clear-input-btn');
    const judgingIndicator = document.getElementById('judging-indicator');
    const judgingMessage = document.getElementById('judging-message');
    const resultContainer = document.getElementById('result-container');
    const resultBanner = document.getElementById('result-banner');
    const resultVerdictIcon = document.getElementById('result-verdict-icon');
    const resultVerdictTitle = document.getElementById('result-verdict-title');
    const resultEngineBadge = document.getElementById('result-engine-badge');
    const resultLatencyBadge = document.getElementById('result-latency-badge');
    const resultTokensBadge = document.getElementById('result-tokens-badge');
    const resultCardsGrid = document.getElementById('result-cards-grid');
    const resultMetricsRow = document.getElementById('result-metrics-row');
    const resultJson = document.getElementById('result-json');
    const prevGameBtn = document.getElementById('prev-game-btn');
    const nextGameBtn = document.getElementById('next-game-btn');
    const modalGameCounter = document.getElementById('modal-game-counter');

    // Friendly Japanese translations for common schema values
    const VALUE_LABELS = {
        // Boss Reaction
        "forgiven": { text: "許された！", badge: "badge-pass", icon: "😇" },
        "promoted": { text: "まさかの昇進！？", badge: "badge-god", icon: "👑" },
        "scolded": { text: "厳重説教", badge: "badge-high", icon: "⚡" },
        "fired": { text: "即日解雇！", badge: "badge-fired", icon: "💥" },

        // Verdicts & Truth
        "absolute_truth": { text: "完全なる真実", badge: "badge-pass", icon: "💎" },
        "mostly_truth": { text: "おおむね真実", badge: "badge-safe", icon: "✅" },
        "suspicious": { text: "極めて疑わしい", badge: "badge-high", icon: "🧐" },
        "blatant_lie": { text: "大嘘・捏造！", badge: "badge-blatant_lie", icon: "🚨" },

        // Levels / Grades
        "god_tier": { text: "神の領域 (God Tier)", badge: "badge-god", icon: "🌟" },
        "legendary": { text: "伝説級 (Legendary)", badge: "badge-legendary", icon: "✨" },
        "high": { text: "高評価 (High)", badge: "badge-safe", icon: "🟢" },
        "medium": { text: "普通 (Medium)", badge: "badge-safe", icon: "⚪" },
        "low": { text: "低 (Low)", badge: "badge-high", icon: "🟡" },

        // Survival / Riddle
        "survived": { text: "生存成功！", badge: "badge-pass", icon: "🎉" },
        "dead": { text: "無念の脱落...", badge: "badge-dead", icon: "💀" },
        "safe": { text: "安全クリア", badge: "badge-safe", icon: "🛡️" },
        "exploded": { text: "大爆発！！", badge: "badge-exploded", icon: "💣" },
        "defused": { text: "解除成功！", badge: "badge-pass", icon: "✂️" },

        // Comedy Dojo
        "ippon": { text: "一本！！！", badge: "badge-god", icon: "💮" },
        "wazaari": { text: "技あり！", badge: "badge-safe", icon: "👏" },
        "warai_nashi": { text: "スベり倒し...", badge: "badge-fired", icon: "🥶" },

        // Salesman
        "purchased_bulk": { text: "爆買い成約！", badge: "badge-god", icon: "💰" },
        "purchased_one": { text: "1つお買い上げ", badge: "badge-pass", icon: "🤝" },
        "interested": { text: "興味津々", badge: "badge-safe", icon: "👀" },
        "police_called": { text: "通報・連行！", badge: "badge-fired", icon: "🚓" },

        // Katsudon
        "full_confession": { text: "涙の完落ち！", badge: "badge-pass", icon: "😭" },
        "eat_katsudon": { text: "カツ丼完食！", badge: "badge-safe", icon: "🍲" },
        "silent": { text: "完全黙秘", badge: "badge-high", icon: "🤐" },
        "lawyer": { text: "弁護士要求", badge: "badge-fired", icon: "⚖️" },

        // Romance / Crush
        "head_over_heels": { text: "脈アリ1000%！", badge: "badge-god", icon: "💘" },
        "crush_confirmed": { text: "両思い確定！", badge: "badge-pass", icon: "💖" },
        "friendly_only": { text: "ただの良き友", badge: "badge-safe", icon: "🍵" },
        "friendzone": { text: "友達止まり", badge: "badge-high", icon: "🧱" },
        "creeped_out": { text: "引かれてる...", badge: "badge-fired", icon: "💔" }
    };

    // Initialize UI
    updateStatsDisplay();

    // Fetch games list
    fetch('/api/games')
        .then(res => res.json())
        .then(data => {
            games = data.games || [];
            document.getElementById('total-games-count').innerText = games.length;
            renderGamesGrid();
        })
        .catch(err => {
            console.error("Failed to load games:", err);
            gamesGrid.innerHTML = `
                <div class="loading-placeholder">
                    <p style="color: #ff4757;">ゲームデータの取得に失敗しました: ${err.message}</p>
                </div>
            `;
        });

    // Check backend health
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            if (data.local_engine && !data.local_engine.connected) {
                console.warn("Local engine not reachable:", data.local_engine);
            }
        })
        .catch(err => console.warn("Health check error:", err));

    // Render Game Cards
    function renderGamesGrid() {
        gamesGrid.innerHTML = '';
        const filterTags = currentFilter === 'all' ? null : currentFilter.split(',');

        games.forEach((game, index) => {
            if (filterTags && !filterTags.includes(game.tag)) {
                return;
            }

            const card = document.createElement('div');
            card.className = 'game-card';
            card.style.setProperty('--card-color', game.badge_color || '#00d2d3');

            const schemaKeys = Object.keys(game.schema || {});
            card.innerHTML = `
                <div class="card-top">
                    <span class="card-emoji">${game.emoji}</span>
                    <span class="card-tag">${game.tag}</span>
                </div>
                <h3 class="card-title">${game.title}</h3>
                <div class="card-subtitle">${game.subtitle}</div>
                <p class="card-desc">${game.description}</p>
                <div class="card-bottom">
                    <span class="card-schema-count">${schemaKeys.length} 判定軸</span>
                    <button class="card-play-btn" data-index="${index}">プレイ ▶</button>
                </div>
            `;

            card.addEventListener('click', () => openGame(index));
            gamesGrid.appendChild(card);
        });
    }

    // Filter tab handler
    filterTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            filterTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentFilter = tab.dataset.filter;
            if (window.soundFX) window.soundFX.click();
            renderGamesGrid();
        });
    });

    // Engine toggle handler
    engineLocalBtn.addEventListener('click', () => {
        currentEngine = 'local';
        engineLocalBtn.classList.add('active');
        engineCloudBtn.classList.remove('active');
        lastEngineLabel.innerText = "Local A100";
        if (window.soundFX) window.soundFX.select();
    });

    engineCloudBtn.addEventListener('click', () => {
        currentEngine = 'official';
        engineCloudBtn.classList.add('active');
        engineLocalBtn.classList.remove('active');
        lastEngineLabel.innerText = "Official Jev";
        if (window.soundFX) window.soundFX.select();
    });

    // Sound toggle handler
    soundBtn.addEventListener('click', () => {
        if (window.soundFX) {
            const enabled = window.soundFX.toggle();
            soundIcon.innerText = enabled ? '🔊' : '🔇';
            if (enabled) window.soundFX.click();
        }
    });

    // Open Game in Modal
    function openGame(index) {
        if (index < 0 || index >= games.length) return;
        currentGameIndex = index;
        const game = games[index];

        modalEmoji.innerText = game.emoji;
        modalTag.innerText = game.tag;
        modalTag.style.color = game.badge_color || 'var(--primary-cyan)';
        modalGameId.innerText = `#${index + 1} / ${games.length}`;
        modalTitle.innerText = game.title;
        modalDesc.innerText = game.description;
        modalScenario.innerText = game.scenario;
        modalInputLabel.innerText = game.input_label || "あなたの回答・行動";
        userTextInput.placeholder = game.input_placeholder || "ここに入力するか、ワンタップボタンを押してください...";
        modalGameCounter.innerText = `${index + 1} / ${games.length}`;

        // Render Presets Chips
        modalPresets.innerHTML = '';
        (game.presets || []).forEach(preset => {
            const chip = document.createElement('button');
            chip.className = 'preset-chip';
            chip.innerText = preset;
            chip.addEventListener('click', () => {
                userTextInput.value = preset;
                if (window.soundFX) window.soundFX.click();
                userTextInput.focus();
            });
            modalPresets.appendChild(chip);
        });

        // Pre-fill with first preset
        if (game.presets && game.presets.length > 0) {
            userTextInput.value = game.presets[0];
        } else {
            userTextInput.value = '';
        }

        // Reset results & state
        resultContainer.classList.add('hidden');
        judgingIndicator.classList.add('hidden');

        // Show modal
        modal.classList.remove('hidden');
        if (window.soundFX) window.soundFX.select();
    }

    // Close Modal
    function closeModal() {
        modal.classList.add('hidden');
        if (window.soundFX) window.soundFX.click();
    }

    modalCloseBtn.addEventListener('click', closeModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    // Prev / Next Navigation
    prevGameBtn.addEventListener('click', () => {
        const nextIdx = (currentGameIndex - 1 + games.length) % games.length;
        openGame(nextIdx);
    });

    nextGameBtn.addEventListener('click', () => {
        const nextIdx = (currentGameIndex + 1) % games.length;
        openGame(nextIdx);
    });

    // Clear input
    clearInputBtn.addEventListener('click', () => {
        userTextInput.value = '';
        userTextInput.focus();
        if (window.soundFX) window.soundFX.click();
    });

    // Play Submit
    playSubmitBtn.addEventListener('click', async () => {
        const inputVal = userTextInput.value.trim();
        if (!inputVal) {
            alert("回答またはアクションを入力してください！");
            userTextInput.focus();
            return;
        }

        const game = games[currentGameIndex];
        judgingIndicator.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        playSubmitBtn.disabled = true;

        if (window.soundFX) window.soundFX.startPlay();

        judgingMessage.innerText = currentEngine === 'official'
            ? 'Official TypeSafe System One Jev クラウド査定中...'
            : 'DeepSeek-V4.1 (4x A100) 非自己回帰意味空間ジャッジ中...';

        try {
            const resp = await fetch('/api/play', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    game_id: game.id,
                    user_input: inputVal,
                    engine: currentEngine
                })
            });

            const data = await resp.json();
            if (!resp.ok) {
                throw new Error(data.detail || "評価エラーが発生しました");
            }

            renderEvaluationResult(data);
            recordPlayStat(data.elapsed_ms);

        } catch (err) {
            alert("エラー: " + err.message);
            if (window.soundFX) window.soundFX.failure();
        } finally {
            judgingIndicator.classList.add('hidden');
            playSubmitBtn.disabled = false;
        }
    });

    // Render Evaluation Result
    function renderEvaluationResult(data) {
        resultContainer.classList.remove('hidden');
        resultEngineBadge.innerText = data.engine || currentEngine;
        resultLatencyBadge.innerText = `⚡ ${data.elapsed_ms}ms`;
        resultTokensBadge.innerText = `${data.tokens || 0} tokens`;

        const resultObj = data.result || {};
        resultCardsGrid.innerHTML = '';

        let hasFiredOrDead = false;
        let hasGodTierOrWin = false;

        // Populate Attribute Cards
        Object.entries(resultObj).forEach(([key, val]) => {
            const strVal = String(val);
            const lookup = VALUE_LABELS[strVal] || null;

            if (strVal.includes('fired') || strVal.includes('dead') || strVal.includes('blatant') || strVal.includes('fail') || strVal.includes('exploded') || strVal.includes('guilty')) {
                hasFiredOrDead = true;
            }
            if (strVal.includes('god') || strVal.includes('promoted') || strVal.includes('survived') || strVal.includes('ippon') || strVal.includes('legendary') || strVal.includes('1000%')) {
                hasGodTierOrWin = true;
            }

            const card = document.createElement('div');
            card.className = 'result-attribute-card';

            let valHtml = '';
            if (lookup) {
                valHtml = `<span class="attr-badge ${lookup.badge}">${lookup.icon} ${lookup.text}</span>`;
            } else if (typeof val === 'boolean') {
                valHtml = val 
                    ? `<span class="attr-badge badge-pass">✅ TRUE</span>`
                    : `<span class="attr-badge badge-fired">❌ FALSE</span>`;
            } else {
                valHtml = `<span class="attr-badge badge-safe">${formatValueLabel(strVal)}</span>`;
            }

            card.innerHTML = `
                <div class="attr-key">${formatKeyLabel(key)}</div>
                <div class="attr-val">${valHtml}</div>
            `;
            resultCardsGrid.appendChild(card);
        });

        // Banner Outcome Verdict
        if (hasGodTierOrWin) {
            resultVerdictIcon.innerText = "🌟";
            resultVerdictTitle.innerText = "超絶判定・神クリア達成！！";
            resultBanner.style.color = "var(--accent-gold)";
            if (window.soundFX) window.soundFX.critical();
        } else if (hasFiredOrDead) {
            resultVerdictIcon.innerText = "💥";
            resultVerdictTitle.innerText = "無残な結末... 判定不合格！";
            resultBanner.style.color = "var(--accent-red)";
            if (window.soundFX) window.soundFX.failure();
        } else {
            resultVerdictIcon.innerText = "🎯";
            resultVerdictTitle.innerText = "JEV 査定完了";
            resultBanner.style.color = "var(--primary-cyan)";
            if (window.soundFX) window.soundFX.success();
        }

        // Metrics Chips
        resultMetricsRow.innerHTML = '';
        const metrics = data.metrics || {};
        if (metrics.scoring_ms) {
            resultMetricsRow.innerHTML += `<div class="metric-chip">Scoring Latency: <strong>${Math.round(metrics.scoring_ms)}ms</strong></div>`;
        }
        if (metrics.prefill_ms) {
            resultMetricsRow.innerHTML += `<div class="metric-chip">Prefill: <strong>${Math.round(metrics.prefill_ms)}ms</strong></div>`;
        }
        if (metrics.tokens_saved) {
            resultMetricsRow.innerHTML += `<div class="metric-chip">Prefix Reused: <strong>+${metrics.tokens_saved} tokens</strong></div>`;
        }
        if (metrics.tok_s) {
            resultMetricsRow.innerHTML += `<div class="metric-chip">Throughput: <strong>${Math.round(metrics.tok_s)} tok/s</strong></div>`;
        }

        // Raw JSON Display
        resultJson.innerText = JSON.stringify(data, null, 2);

        // Scroll result into view smoothly
        setTimeout(() => {
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
    }

    function formatKeyLabel(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    function formatValueLabel(val) {
        return val.replace(/_/g, ' ');
    }

    // Stats helper
    function recordPlayStat(latencyMs) {
        stats.plays += 1;
        stats.totalMs += latencyMs;
        localStorage.setItem('jev_arcade_stats', JSON.stringify(stats));
        updateStatsDisplay();
    }

    function updateStatsDisplay() {
        playCounter.innerText = stats.plays;
        if (stats.plays > 0) {
            const avg = Math.round(stats.totalMs / stats.plays);
            avgLatency.innerText = `${avg}ms`;
        } else {
            avgLatency.innerText = '--';
        }
    }

    // Keyboard navigation
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            closeModal();
        } else if (e.key === 'ArrowLeft' && !modal.classList.contains('hidden')) {
            prevGameBtn.click();
        } else if (e.key === 'ArrowRight' && !modal.classList.contains('hidden')) {
            nextGameBtn.click();
        }
    });

})();

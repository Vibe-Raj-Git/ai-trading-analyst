// main.js - Enhanced for full trading analysis features with ATR support

// State management
let currentAnalysis = null;
let currentSymbol = '';
let currentMode = '';

// DOM Elements
const analyzeBtn = document.getElementById('analyzeBtn');
const symbolInput = document.getElementById('symbol');
const modeSelect = document.getElementById('mode');
const loadingOverlay = document.getElementById('loadingOverlay');
const quickActionCard = document.getElementById('quickActionCard');
const trainerExplanation = document.getElementById('trainerExplanation');
const rawAnalysis = document.getElementById('rawAnalysis');
const rawAnalysisContainer = document.getElementById('rawAnalysisContainer');
const exportPdfBtn = document.getElementById('exportPdfBtn');
const analysisInfo = document.getElementById('analysisInfo');
const strategyInfo = document.getElementById('strategyInfo');
const regimeBadges = document.getElementById('regimeBadges');
const foContext = document.getElementById('foContext');
const foMetrics = document.getElementById('foMetrics');
const foSignals = document.getElementById('foSignals');
const foDecision = document.getElementById('foDecision');

// Key Metrics Elements
const keyMetrics = document.getElementById('keyMetrics');
const marketStageDisplay = document.getElementById('marketStageDisplay');
const trendStrengthDisplay = document.getElementById('trendStrengthDisplay');
const atrDisplay = document.getElementById('atrDisplay');
const atrValueSpan = document.getElementById('atrValue');
const atrPercentSpan = document.getElementById('atrPercent');
const volatilityBadge = document.getElementById('volatilityBadge');
const weeklyAtrSpan = document.getElementById('weeklyAtr');
const monthlyAtrSpan = document.getElementById('monthlyAtr');
const atrHistoryDiv = document.getElementById('atrHistory');

// Theme toggle
function toggleTheme() {
    const isDark = document.body.getAttribute('data-theme') === 'dark';
    document.body.setAttribute('data-theme', isDark ? 'light' : 'dark');
    const themeText = document.getElementById('themeText');
    if (themeText) themeText.textContent = isDark ? 'Dark' : 'Light';
}

// Run analysis when Enter key pressed
if (symbolInput) {
    symbolInput.addEventListener('input', function() {
        this.value = this.value.toUpperCase();
    });
    
    symbolInput.addEventListener('paste', function(e) {
        setTimeout(() => {
            this.value = this.value.toUpperCase();
        }, 0);
    });
    
    symbolInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') runAnalysis();
    });
    
    symbolInput.value = symbolInput.value.toUpperCase();
}

// Mode info display
const modeInfo = document.getElementById('modeInfo');
if (modeSelect && modeInfo) {
    const modeDescriptions = {
        'Intraday': '⚡ Intraday (Triple-Screen): Daily → 30m → 5m\n• Focus: Momentum breakouts with RVOL > 1.2\n• Range trades only when Daily is Range/SmartRange',
        'Swing': '🔄 Swing (Triple-Screen): Weekly → Daily → 4H\n• Gate 1: Weekly RetailChop = No trades\n• Gate 2: 4H RetailChop = No entries\n• Clean trend = Trend-Following + Pullback',
        'Positional': '📊 Positional (Triple-Screen): Monthly → Weekly → Daily\n• Monthly master bias determines direction\n• SmartRange = breakout positioning\n• Range = mean-reversion only',
        'F&O': '🎯 F&O (Triple-Screen): Daily → 30m → 5m\n• FO metrics (IV, PCR, Greeks) tune conviction\n• Daily Darvas box defines cash market range\n• Gamma exposure adjusts position sizing',
        'Investing': '💰 Investing (Triple-Screen): Quarterly → Monthly → Weekly\n• Quarterly master bias = accumulate near discount\n• Monthly Darvas box defines value zones\n• Range = selective value entries'
    };
    
    modeSelect.addEventListener('change', function() {
        const mode = this.value;
        modeInfo.innerHTML = modeDescriptions[mode] || 'Select a mode to see explanation';
        modeInfo.style.whiteSpace = 'pre-line';
    });
    
    modeSelect.dispatchEvent(new Event('change'));
}

// Main analysis function
async function runAnalysis() {
    if (!symbolInput) {
        showError('Symbol input not found');
        return;
    }
    
    const symbol = symbolInput.value.trim().toUpperCase();
    const mode = modeSelect ? modeSelect.value : 'Intraday';
    
    if (!symbol) {
        showError('Please enter a valid stock or index symbol.');
        return;
    }
    
    currentSymbol = symbol;
    currentMode = mode;
    
    if (loadingOverlay) loadingOverlay.style.display = 'flex';
    if (quickActionCard) quickActionCard.style.display = 'none';
    if (exportPdfBtn) exportPdfBtn.style.display = 'none';
    if (rawAnalysisContainer) rawAnalysisContainer.style.display = 'none';
    if (foContext) foContext.style.display = 'none';
    if (keyMetrics) keyMetrics.style.display = 'none';
    if (atrHistoryDiv) atrHistoryDiv.style.display = 'none';
    
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({symbol: symbol, mode: mode})
        });
        
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error || 'Analysis failed');
        
        currentAnalysis = data;
        
        const analysisSymbol = document.getElementById('analysisSymbol');
        const analysisMode = document.getElementById('analysisMode');
        const analysisTimestamp = document.getElementById('analysisTimestamp');
        
        if (analysisSymbol) analysisSymbol.textContent = data.symbol;
        if (analysisMode) analysisMode.textContent = data.mode;
        if (analysisTimestamp) analysisTimestamp.textContent = data.timestamp;
        if (analysisInfo) analysisInfo.style.display = 'block';
        
        displayQuickAction(data.quick_action);
        displayRegimeBadges(data.regimes);
        displayKeyMetrics(data);
        displayTrainerExplanation(data.explanation);
        
        if (data.mode === 'F&O' && data.fo_context && Object.keys(data.fo_context).length > 0) {
            displayFOContext(data.fo_context);
        }
        
        displayRawAnalysis(data.raw_analysis);
        
        if (exportPdfBtn) exportPdfBtn.style.display = 'inline-block';
        
        const quickCard = document.getElementById('quickActionCard');
        if (quickCard) quickCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message);
    } finally {
        if (loadingOverlay) loadingOverlay.style.display = 'none';
    }
}

// Display key metrics dashboard with enhanced ATR support
function displayKeyMetrics(data) {
    if (!keyMetrics) return;
    
    keyMetrics.style.display = 'grid';
    
    // Market Stage
    const marketStage = data.regimes?.Market_Stage || 
                       (data.regimes?.Trend_Regime === 'Bullish' ? 'Advancing' :
                        data.regimes?.Trend_Regime === 'Bearish' ? 'Declining' : 'Accumulation');
    if (marketStageDisplay) marketStageDisplay.textContent = marketStage;
    
    // Trend Strength - Use Trend Score from regimes if available
    let trendStrength = 'Moderate';
    let trendScore = data.regimes?.Trend_Score;
    if (trendScore !== undefined && trendScore !== null) {
        if (trendScore >= 7) trendStrength = 'Strong';
        else if (trendScore >= 4) trendStrength = 'Moderate';
        else trendStrength = 'Weak';
    } else {
        const adxMatch = data.raw_analysis?.match(/ADX14[:\s]+(\d+\.?\d*)/i);
        if (adxMatch) {
            const adx = parseFloat(adxMatch[1]);
            if (adx > 40) trendStrength = 'Very Strong';
            else if (adx > 25) trendStrength = 'Strong';
            else if (adx < 20) trendStrength = 'Weak';
        }
    }
    if (trendStrengthDisplay) trendStrengthDisplay.textContent = trendStrength;
    
    displayATRMetrics(data);
}

// Enhanced ATR Display
function displayATRMetrics(data) {
    if (data.atr_metrics && data.atr_metrics.atr_display && data.atr_metrics.atr_display !== '--') {
        if (atrValueSpan) {
            atrValueSpan.textContent = data.atr_metrics.atr_display;
        }
        
        if (atrPercentSpan && data.atr_metrics.atr_percent_display) {
            atrPercentSpan.textContent = data.atr_metrics.atr_percent_display;
            atrPercentSpan.style.display = 'inline';
        } else if (atrPercentSpan) {
            atrPercentSpan.style.display = 'none';
        }
        
        if (volatilityBadge) {
            const regime = data.atr_metrics.volatility_regime;
            if (regime === 'low') {
                volatilityBadge.className = 'volatility-badge volatility-low';
                volatilityBadge.innerHTML = '📊 Low Volatility';
                volatilityBadge.setAttribute('data-tooltip', 'Low volatility environment - tighter stops appropriate');
            } else if (regime === 'high') {
                volatilityBadge.className = 'volatility-badge volatility-high';
                volatilityBadge.innerHTML = '⚠️ High Volatility - Widen stops, reduce size';
                volatilityBadge.setAttribute('data-tooltip', 'High volatility environment - increase stop distance, reduce position size by 50%');
            } else {
                volatilityBadge.className = 'volatility-badge volatility-normal';
                volatilityBadge.innerHTML = '📈 Normal Volatility';
                volatilityBadge.setAttribute('data-tooltip', 'Normal volatility environment - standard position sizing');
            }
            volatilityBadge.style.display = 'inline-flex';
        }
        
        if (atrHistoryDiv && (data.atr_metrics.weekly_atr || data.atr_metrics.monthly_atr)) {
            if (weeklyAtrSpan && data.atr_metrics.weekly_atr) {
                weeklyAtrSpan.textContent = data.atr_metrics.weekly_atr;
            }
            if (monthlyAtrSpan && data.atr_metrics.monthly_atr) {
                monthlyAtrSpan.textContent = data.atr_metrics.monthly_atr;
            }
            atrHistoryDiv.style.display = 'flex';
        }
        
    } else {
        const atrMatch = data.raw_analysis?.match(/ATR[:\s]+(\d+\.?\d*)/i);
        if (atrMatch && atrValueSpan) {
            atrValueSpan.textContent = atrMatch[1];
            
            const priceMatch = data.raw_analysis?.match(/Current Price[:\s]+(\d+\.?\d*)/i) ||
                              data.raw_analysis?.match(/Close[:\s]+(\d+\.?\d*)/i);
            if (priceMatch && atrPercentSpan) {
                const atr = parseFloat(atrMatch[1]);
                const price = parseFloat(priceMatch[1]);
                const percent = (atr / price * 100).toFixed(2);
                atrPercentSpan.textContent = `(${percent}%)`;
                atrPercentSpan.style.display = 'inline';
                
                if (volatilityBadge) {
                    if (percent < 1) {
                        volatilityBadge.className = 'volatility-badge volatility-low';
                        volatilityBadge.innerHTML = '📊 Low Volatility';
                    } else if (percent > 2) {
                        volatilityBadge.className = 'volatility-badge volatility-high';
                        volatilityBadge.innerHTML = '⚠️ High Volatility';
                    } else {
                        volatilityBadge.className = 'volatility-badge volatility-normal';
                        volatilityBadge.innerHTML = '📈 Normal Volatility';
                    }
                    volatilityBadge.style.display = 'inline-flex';
                }
            } else if (atrPercentSpan) {
                atrPercentSpan.style.display = 'none';
            }
        } else if (atrValueSpan) {
            atrValueSpan.textContent = '--';
            if (volatilityBadge) volatilityBadge.style.display = 'none';
        }
    }
    
    if (atrDisplay && atrValueSpan) {
        atrDisplay.innerHTML = atrValueSpan.innerHTML;
    }
}

// Display quick action
function displayQuickAction(quickAction) {
    const quickCard = document.getElementById('quickActionCard');
    if (!quickCard) return;
    
    let className = '', iconClass = '';
    
    switch(quickAction.bias) {
        case 'BUY': className = 'quick-action-buy'; iconClass = 'fa-arrow-up'; break;
        case 'RANGE-BUY': className = 'quick-action-range-buy'; iconClass = 'fa-chart-line'; break;
        case 'SELL': className = 'quick-action-sell'; iconClass = 'fa-arrow-down'; break;
        case 'RANGE-SELL': className = 'quick-action-range-sell'; iconClass = 'fa-chart-line'; break;
        default: className = 'quick-action-wait'; iconClass = 'fa-hourglass-half';
    }
    
    quickCard.className = `quick-action-card ${className}`;
    quickCard.innerHTML = `
        <i class="fas ${iconClass}" style="font-size: 3rem;"></i>
        <h2 class="mb-2 mt-3">${quickAction.bias}</h2>
        <p class="mb-0 fs-5">${quickAction.message}</p>
    `;
    quickCard.style.display = 'block';
}

// Display regime badges
function displayRegimeBadges(regimes) {
    const regimeDiv = document.getElementById('regimeBadges');
    if (!regimeDiv || !regimes) return;
    
    const getRegimeClass = (regime) => {
        if (!regime) return '';
        const r = regime.toLowerCase();
        if (r === 'bullish') return 'regime-bullish';
        if (r === 'bearish') return 'regime-bearish';
        return 'regime-range';
    };
    
    regimeDiv.innerHTML = `
        <span class="regime-badge ${getRegimeClass(regimes.Trend_Regime)}">
            <i class="fas fa-chart-line"></i> Trend: ${regimes.Trend_Regime || 'N/A'}
        </span>
        <span class="regime-badge ${getRegimeClass(regimes.Setup_Regime)}">
            <i class="fas fa-chart-bar"></i> Setup: ${regimes.Setup_Regime || 'N/A'}
        </span>
        <span class="regime-badge ${getRegimeClass(regimes.Entry_Regime)}">
            <i class="fas fa-chart-simple"></i> Entry: ${regimes.Entry_Regime || 'N/A'}
        </span>
    `;
}

// Display F&O context
function displayFOContext(foContextData) {
    const foDiv = document.getElementById('foContext');
    const foMetricsDiv = document.getElementById('foMetrics');
    const foSignalsDiv = document.getElementById('foSignals');
    const foDecisionDiv = document.getElementById('foDecision');
    
    if (!foDiv || !foMetricsDiv) return;
    
    // Update IV displays
    const ivCallEl = document.getElementById('fo_iv_call');
    const ivPutEl = document.getElementById('fo_iv_put');
    if (ivCallEl) ivCallEl.textContent = foContextData.iv_call?.toFixed(2) || '--';
    if (ivPutEl) ivPutEl.textContent = foContextData.iv_put?.toFixed(2) || '--';
    
    // Update PCR
    const pcrEl = document.getElementById('fo_pcr');
    if (pcrEl) pcrEl.textContent = foContextData.pcr?.toFixed(2) || '--';
    
    // Update Greeks
    const callDeltaEl = document.getElementById('fo_call_delta');
    const putDeltaEl = document.getElementById('fo_put_delta');
    if (callDeltaEl) callDeltaEl.textContent = foContextData.call_delta?.toFixed(3) || '--';
    if (putDeltaEl) putDeltaEl.textContent = foContextData.put_delta?.toFixed(3) || '--';
    
    const callGammaEl = document.getElementById('fo_call_gamma');
    const putGammaEl = document.getElementById('fo_put_gamma');
    if (callGammaEl) callGammaEl.textContent = foContextData.call_gamma?.toFixed(4) || '--';
    if (putGammaEl) putGammaEl.textContent = foContextData.put_gamma?.toFixed(4) || '--';
    
    const callVegaEl = document.getElementById('fo_call_vega');
    const putVegaEl = document.getElementById('fo_put_vega');
    if (callVegaEl) callVegaEl.textContent = foContextData.call_vega?.toFixed(2) || '--';
    if (putVegaEl) putVegaEl.textContent = foContextData.put_vega?.toFixed(2) || '--';
    
    // Update OI
    const oiTrendEl = document.getElementById('fo_oi_trend');
    const callOiEl = document.getElementById('fo_call_oi');
    const putOiEl = document.getElementById('fo_put_oi');
    if (oiTrendEl) oiTrendEl.textContent = foContextData.oi_trend || '--';
    if (callOiEl) callOiEl.textContent = foContextData.total_call_oi?.toLocaleString() || '--';
    if (putOiEl) putOiEl.textContent = foContextData.total_put_oi?.toLocaleString() || '--';
    
    // Update Futures
    const futuresStateEl = document.getElementById('fo_futures_state');
    const futuresPriceEl = document.getElementById('fo_futures_price');
    if (futuresStateEl) futuresStateEl.textContent = foContextData.futures_state || '--';
    if (futuresPriceEl) futuresPriceEl.textContent = foContextData.futures_price?.toFixed(2) || '--';
    
    // Update FO Decision - FIXED VERSION
    const decisionBiasEl = document.getElementById('fo_decision_bias');
    const decisionConvictionEl = document.getElementById('fo_decision_conviction');
    const decisionRiskEl = document.getElementById('fo_decision_risk');
    
    if (decisionBiasEl && foContextData.fo_bias) {
        decisionBiasEl.textContent = foContextData.fo_bias;
        
        if (decisionConvictionEl) {
            decisionConvictionEl.textContent = foContextData.fo_conviction || '--';
        }
        
        if (decisionRiskEl) {
            // Check for no_trade condition
            if (foContextData.fo_no_trade === true || foContextData.fo_bias === 'no_trade') {
                decisionRiskEl.textContent = 'No Trade';
            } else if (foContextData.fo_risk_profile) {
                decisionRiskEl.textContent = foContextData.fo_risk_profile;
            } else {
                // Fallback based on conviction
                const conviction = foContextData.fo_conviction || '';
                if (conviction === 'HIGH') decisionRiskEl.textContent = 'Aggressive';
                else if (conviction === 'MEDIUM') decisionRiskEl.textContent = 'Moderate';
                else decisionRiskEl.textContent = 'Conservative';
            }
        }
        
        if (foDecisionDiv) foDecisionDiv.style.display = 'block';
    }
    
    foDiv.style.display = 'block';
}

// Display trainer explanation
function displayTrainerExplanation(explanation) {
    const explanationDiv = document.getElementById('trainerExplanation');
    if (!explanationDiv) return;
    
    let formatted = explanation || '';
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formatted = formatted.replace(/^### (.*$)/gm, '<h3>$1</h3>');
    formatted = formatted.replace(/^## (.*$)/gm, '<h2>$1</h2>');
    formatted = formatted.replace(/^# (.*$)/gm, '<h1>$1</h1>');
    formatted = formatted.replace(/^- (.*$)/gm, '<li>$1</li>');
    formatted = formatted.replace(/^• (.*$)/gm, '<li>$1</li>');
    formatted = formatted.replace(/(<li>.*?<\/li>\n?)+/g, '<ul>$&</ul>');
    formatted = formatted.replace(/^\d+\. (.*$)/gm, '<li>$1</li>');
    formatted = formatted.replace(/\n\n/g, '</p><p>');
    formatted = formatted.replace(/\n/g, '<br>');
    formatted = formatted.replace(/\b\d{3,}\.\d{2}\b/g, match => 
        `<span class="price-level" onclick="highlightLevel('${match}')">${match}</span>`);
    
    if (formatted && !formatted.startsWith('<')) formatted = '<p>' + formatted + '</p>';
    explanationDiv.innerHTML = formatted || '<p>Analysis complete. No detailed explanation available.</p>';
    addInteractiveLevels();
}

// Add interactive price levels
function addInteractiveLevels() {
    const explanationDiv = document.getElementById('trainerExplanation');
    if (!explanationDiv) return;
    
    const priceRegex = /\b\d{3,}\.\d{2}\b/g;
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = explanationDiv.innerHTML;
    
    function processNode(node) {
        if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent;
            const newText = text.replace(priceRegex, (match) => {
                return `<span class="price-level" onclick="highlightLevel('${match}')" title="Click to highlight this level">${match}</span>`;
            });
            if (newText !== text) {
                const span = document.createElement('span');
                span.innerHTML = newText;
                node.parentNode.replaceChild(span, node);
            }
        } else if (node.nodeType === Node.ELEMENT_NODE && !['SCRIPT', 'STYLE', 'A'].includes(node.tagName)) {
            Array.from(node.childNodes).forEach(processNode);
        }
    }
    
    Array.from(tempDiv.childNodes).forEach(processNode);
    explanationDiv.innerHTML = tempDiv.innerHTML;
}

// Highlight price level
function highlightLevel(price) {
    document.querySelectorAll('.price-highlight').forEach(el => el.classList.remove('price-highlight'));
    document.querySelectorAll('.price-level').forEach(span => {
        if (span.textContent === price) {
            span.classList.add('price-highlight');
            span.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
}

// Display raw analysis
function displayRawAnalysis(rawText) {
    const rawDiv = document.getElementById('rawAnalysis');
    if (!rawDiv) return;
    
    let cleaned = rawText || '';
    const jsonIndex = cleaned.lastIndexOf('STRATEGIES_JSON');
    if (jsonIndex !== -1) cleaned = cleaned.substring(0, jsonIndex);
    
    cleaned = cleaned.replace(/\b(\d+\.\d{2})\b/g, '<span class="raw-number">$1</span>');
    cleaned = cleaned.replace(/\b(\d{3,})\b/g, '<span class="raw-number">$1</span>');
    
    const indicators = ['RSI', 'EMA', 'MACD', 'ADX', 'ATR', 'BB', 'VWAP', 'OBV', 'MFI', 'RVOL', 'PCR', 'IV'];
    indicators.forEach(ind => {
        cleaned = cleaned.replace(new RegExp(`\\b${ind}\\b`, 'gi'), '<span class="raw-indicator">$&</span>');
    });
    
    rawDiv.innerHTML = cleaned;
}

// Toggle raw analysis
function toggleRawAnalysis(event) {
    const container = document.getElementById('rawAnalysisContainer');
    const btn = event?.target?.closest?.('button');
    if (!container) return;
    
    if (container.style.display === 'none') {
        container.style.display = 'block';
        if (btn) btn.innerHTML = '<i class="fas fa-chevron-up"></i> Hide';
    } else {
        container.style.display = 'none';
        if (btn) btn.innerHTML = '<i class="fas fa-chevron-down"></i> Show';
    }
}

// Export PDF
async function exportPDF() {
    if (!currentAnalysis) {
        showError('No analysis to export');
        return;
    }
    
    try {
        const response = await fetch('/export-pdf', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                symbol: currentSymbol,
                mode: currentMode,
                explanation: currentAnalysis.explanation,
                quick_action: currentAnalysis.quick_action,
                strategies: {}
            })
        });
        
        if (!response.ok) throw new Error('PDF generation failed');
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${currentSymbol}_analysis.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showSuccess('PDF downloaded successfully!');
    } catch (error) {
        console.error('PDF export error:', error);
        showError('Failed to export PDF');
    }
}

// Copy to clipboard
function copyToClipboard(event) {
    const trainerDiv = document.getElementById('trainerExplanation');
    if (!trainerDiv) return;
    
    const text = trainerDiv.innerText;
    navigator.clipboard.writeText(text).then(() => {
        const btn = event?.target?.closest?.('button');
        if (btn) {
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            setTimeout(() => btn.innerHTML = originalText, 2000);
        } else {
            showSuccess('Analysis copied to clipboard!');
        }
    }).catch(() => showError('Failed to copy to clipboard'));
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    errorDiv.style.zIndex = '10000';
    errorDiv.style.minWidth = '300px';
    errorDiv.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i> ${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
    document.body.appendChild(errorDiv);
    setTimeout(() => errorDiv.remove(), 5000);
}

// Show success message
function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'alert alert-success alert-dismissible fade show position-fixed top-0 start-50 translate-middle-x mt-3';
    successDiv.style.zIndex = '10000';
    successDiv.style.minWidth = '300px';
    successDiv.innerHTML = `<i class="fas fa-check-circle me-2"></i> ${message}<button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
    document.body.appendChild(successDiv);
    setTimeout(() => successDiv.remove(), 3000);
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runAnalysis();
    }
    if (e.key === 'Escape') {
        document.querySelectorAll('.level-tooltip').forEach(t => t.remove());
    }
});

// Add dynamic styles if not present
if (!document.querySelector('#dynamic-styles')) {
    const style = document.createElement('style');
    style.id = 'dynamic-styles';
    style.textContent = `
        .price-level { background: linear-gradient(135deg, #ff6b35, #ff4500); border-radius: 8px; padding: 4px 12px; cursor: pointer; font-weight: bold; color: white; transition: all 0.2s; display: inline-block; margin: 2px; box-shadow: 0 2px 5px rgba(0,0,0,0.2); }
        .price-level:hover { transform: scale(1.1) translateY(-2px); box-shadow: 0 5px 15px rgba(255,107,53,0.4); }
        .price-highlight { background: linear-gradient(135deg, #ffd93d, #ffc107); color: #333; animation: highlightPulse 0.5s ease; box-shadow: 0 0 10px rgba(255,217,61,0.5); }
        @keyframes highlightPulse { 0% { transform: scale(1); } 50% { transform: scale(1.15); } 100% { transform: scale(1); } }
        .raw-number { color: #4ec9b0; font-weight: bold; text-shadow: 0 0 3px rgba(78,201,176,0.3); }
        .raw-indicator { color: #569cd6; font-weight: bold; text-shadow: 0 0 3px rgba(86,156,214,0.3); }
        .regime-badge { display: inline-flex; align-items: center; gap: 8px; padding: 8px 20px; border-radius: 40px; font-size: 0.85rem; font-weight: 700; transition: all 0.3s; backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.2); }
        .regime-badge:hover { transform: translateY(-2px) scale(1.05); }
        .regime-bullish { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .regime-bearish { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .regime-range { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: white; }
        .quick-action-card { border-radius: 20px; padding: 28px; margin-bottom: 28px; text-align: center; animation: slideIn 0.5s ease; position: relative; overflow: hidden; }
        .quick-action-buy { background: linear-gradient(135deg, #10b981, #059669); color: white; }
        .quick-action-sell { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
        .quick-action-range-buy { background: linear-gradient(135deg, #34d399, #10b981); color: #064e3b; }
        .quick-action-range-sell { background: linear-gradient(135deg, #f87171, #ef4444); color: #7f1a1a; }
        .quick-action-wait { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #78350f; }
        @keyframes slideIn { from { opacity: 0; transform: translateY(-30px); } to { opacity: 1; transform: translateY(0); } }
        .fo-card { background: var(--gradient-card); border-left: 4px solid #ff6b35; border-radius: 16px; padding: 20px; transition: all 0.3s; }
        .fo-card:hover { transform: translateY(-2px); box-shadow: var(--neon-glow); }
        .trainer-content { max-height: 700px; overflow-y: auto; padding: 20px; line-height: 1.8; }
        .trainer-content h1, .trainer-content h2, .trainer-content h3 { margin-top: 24px; margin-bottom: 16px; font-weight: 700; }
        .trainer-content h1 { background: var(--gradient-primary); -webkit-background-clip: text; background-clip: text; color: transparent; font-size: 2rem; }
        .trainer-content h2 { color: #ff6b35; font-size: 1.6rem; border-left: 4px solid #ff6b35; padding-left: 16px; }
        .trainer-content h3 { color: #ffd93d; font-size: 1.3rem; }
        .trainer-content strong { color: #ff6b35; }
        .raw-analysis { background: linear-gradient(135deg, #1a1a2e, #0a0a2a); color: #00ff9d; padding: 20px; border-radius: 16px; font-family: 'Fira Code', monospace; font-size: 0.85rem; max-height: 500px; overflow-y: auto; white-space: pre-wrap; border: 1px solid rgba(0,255,157,0.3); }
        .metrics-grid .card h3 { font-size: 2rem; font-weight: 800; background: var(--gradient-primary); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .volatility-badge { font-size: 0.7rem; font-weight: 600; margin-top: 8px; display: inline-block; padding: 4px 12px; border-radius: 20px; }
        .volatility-low { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
        .volatility-normal { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
        .volatility-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    `;
    document.head.appendChild(style);
}

// Export functions for global access
window.runAnalysis = runAnalysis;
window.displayFOContext = displayFOContext;  // ✅ ADD THIS LINE
window.toggleRawAnalysis = toggleRawAnalysis;
window.exportPDF = exportPDF;
window.copyToClipboard = copyToClipboard;
window.highlightLevel = highlightLevel;
window.toggleTheme = toggleTheme;
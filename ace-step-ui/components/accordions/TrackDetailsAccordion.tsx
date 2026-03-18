import React from 'react';
import { ChevronDown, FileText } from 'lucide-react';
import { useI18n } from '../../context/I18nContext';
import { VOCAL_LANGUAGE_KEYS } from '../../utils/constants';
import { LyricsSection } from '../sections/LyricsSection';
import { StyleSection } from '../sections/StyleSection';
import { MusicParametersSection } from '../sections/MusicParametersSection';

interface TrackDetailsAccordionProps {
    showTrackDetails: boolean;
    setShowTrackDetails: (val: boolean) => void;
    instrumental: boolean;
    setInstrumental: (val: boolean) => void;
    vocalLanguage: string;
    setVocalLanguage: (val: string) => void;
    vocalGender: string;
    setVocalGender: (val: string) => void;
    title: string;
    setTitle: (val: string) => void;

    // Lyrics Props
    showLyricsSub: boolean;
    setShowLyricsSub: (val: boolean) => void;
    lyrics: string;
    setLyrics: React.Dispatch<React.SetStateAction<string>>;
    lyricsRef: React.RefObject<HTMLDivElement>;
    lyricsHeight: number;
    startResizing: (e: React.MouseEvent) => void;
    isFormattingLyrics: boolean;

    // Style Props
    showStyleSub: boolean;
    setShowStyleSub: (val: boolean) => void;
    style: string;
    setStyle: React.Dispatch<React.SetStateAction<string>>;
    refreshMusicTags: () => void;
    isFormattingStyle: boolean;
    handleFormat: (target: 'lyrics' | 'style') => void;
    styleRef: React.RefObject<HTMLDivElement>;
    styleHeight: number;
    startResizingStyle: (e: React.MouseEvent) => void;
    genreDropdownRef: React.RefObject<HTMLDivElement>;
    showGenreDropdown: boolean;
    setShowGenreDropdown: (val: boolean) => void;
    selectedMainGenre: string;
    setSelectedMainGenre: (val: string) => void;
    selectedSubGenre: string;
    setSelectedSubGenre: (val: string) => void;
    getSubGenreCount: (mainGenre: string) => number;
    genreSearch: string;
    setGenreSearch: (val: string) => void;
    filteredCombinedGenres: { name: string, type: 'main' | 'other' }[];
    subGenreDropdownRef: React.RefObject<HTMLDivElement>;
    showSubGenreDropdown: boolean;
    setShowSubGenreDropdown: (val: boolean) => void;
    filteredSubGenres: string[];
    musicTags: string[];

    // Music Parameters Props
    bpm: number;
    setBpm: (val: number) => void;
    keyScale: string;
    setKeyScale: (val: string) => void;
    timeSignature: string;
    setTimeSignature: (val: string) => void;
    duration: number;
    setDuration: (val: number) => void;
    detectedBpm?: number | null;
    detectedKey?: string | null;
    triggerWord?: string;
    taskType?: string;
    sourceDuration?: number;
    tempoScale?: number;
    effectiveBpm?: number;
    effectiveKeyScale?: string;
}

export const TrackDetailsAccordion: React.FC<TrackDetailsAccordionProps> = ({
    showTrackDetails, setShowTrackDetails,
    instrumental, setInstrumental,
    vocalLanguage, setVocalLanguage,
    vocalGender, setVocalGender,
    title, setTitle,
    showLyricsSub, setShowLyricsSub, lyrics, setLyrics, lyricsRef, lyricsHeight, startResizing, isFormattingLyrics,
    showStyleSub, setShowStyleSub, style, setStyle, refreshMusicTags, isFormattingStyle, handleFormat, styleRef, styleHeight, startResizingStyle, genreDropdownRef, showGenreDropdown, setShowGenreDropdown, selectedMainGenre, setSelectedMainGenre, selectedSubGenre, setSelectedSubGenre, getSubGenreCount, genreSearch, setGenreSearch, filteredCombinedGenres, subGenreDropdownRef, showSubGenreDropdown, setShowSubGenreDropdown, filteredSubGenres, musicTags,
    bpm, setBpm, keyScale, setKeyScale, timeSignature, setTimeSignature, duration, setDuration,
    detectedBpm, detectedKey, triggerWord,
    taskType, sourceDuration, tempoScale,
    effectiveBpm, effectiveKeyScale,
}) => {
    const { t } = useI18n();

    return (
        <div>
            <button
                type="button"
                onClick={() => setShowTrackDetails(!showTrackDetails)}
                className={`w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-suno-card border border-zinc-200 dark:border-white/5 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-50 dark:hover:bg-white/5 transition-colors ${showTrackDetails ? 'rounded-t-xl rounded-b-none border-b-0' : 'rounded-xl'}`}
            >
                <span className="flex items-center gap-2"><FileText size={16} className="text-pink-500" />{t('trackDetails')}</span>
                <ChevronDown size={18} className={`text-pink-500 chevron-icon ${showTrackDetails ? 'rotated' : ''}`} />
            </button>
            {showTrackDetails && (
                <div className="bg-white dark:bg-suno-card rounded-b-xl rounded-t-none border border-t-0 border-zinc-200 dark:border-white/5 p-4 space-y-4">

                    {/* Vocal Language & Gender */}
                    {!instrumental && (
                        <div className="grid grid-cols-2 gap-3 mb-2">
                            <div className="space-y-1.5">
                                <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('vocalLanguageTooltip')}>{t('vocalLanguage')}</label>
                                <select
                                    value={vocalLanguage}
                                    onChange={(e) => setVocalLanguage(e.target.value)}
                                    className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-2 py-2 text-xs text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors cursor-pointer [&>option]:bg-white [&>option]:dark:bg-zinc-800 [&>option]:text-zinc-900 [&>option]:dark:text-white"
                                >
                                    {VOCAL_LANGUAGE_KEYS.map(lang => (
                                        <option key={lang.value} value={lang.value}>{t(lang.key)}</option>
                                    ))}
                                </select>
                            </div>
                            <div className="space-y-1.5">
                                <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('vocalGenderTooltip')}>{t('vocalGender')}</label>
                                <div className="flex items-center gap-2">
                                    <button
                                        type="button"
                                        onClick={() => setVocalGender(vocalGender === 'male' ? '' : 'male')}
                                        className={`flex-1 px-3 py-2 rounded-lg text-xs font-semibold border transition-colors ${vocalGender === 'male' ? 'bg-pink-600 text-white border-pink-600' : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-white/20'}`}
                                    >
                                        {t('male')}
                                    </button>
                                    <button
                                        type="button"
                                        onClick={() => setVocalGender(vocalGender === 'female' ? '' : 'female')}
                                        className={`flex-1 px-3 py-2 rounded-lg text-xs font-semibold border transition-colors ${vocalGender === 'female' ? 'bg-pink-600 text-white border-pink-600' : 'border-zinc-200 dark:border-white/10 text-zinc-600 dark:text-zinc-300 hover:border-zinc-300 dark:hover:border-white/20'}`}
                                    >
                                        {t('female')}
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Title Input */}
                    <div className="space-y-1.5">
                        <label className="text-xs font-medium text-zinc-600 dark:text-zinc-400" title={t('titleTooltip')}>{t('title')}</label>
                        <input
                            type="text"
                            value={title}
                            onChange={(e) => setTitle(e.target.value)}
                            placeholder={t('nameSong')}
                            className="w-full bg-zinc-50 dark:bg-black/20 border border-zinc-200 dark:border-white/10 rounded-lg px-3 py-2 text-sm text-zinc-900 dark:text-white focus:outline-none focus:border-pink-500 dark:focus:border-pink-500 transition-colors"
                        />
                    </div>

                    {/* Instrumental Toggle */}
                    <div className="flex items-center justify-between py-1 border-b border-zinc-200 dark:border-white/5 pb-3">
                        <div>
                            <span className="text-xs font-medium text-zinc-600 dark:text-zinc-400">{t('instrumental')}</span>
                            <p className="text-[11px] text-zinc-400 dark:text-zinc-500">{t('instrumentalTooltip')}</p>
                        </div>
                        <button
                            onClick={() => setInstrumental(!instrumental)}
                            className={`w-10 h-5 rounded-full flex items-center transition-colors duration-200 px-0.5 border border-zinc-200 dark:border-white/5 ${instrumental ? 'bg-pink-600' : 'bg-zinc-300 dark:bg-black/40'} cursor-pointer`}
                        >
                            <div className={`w-4 h-4 rounded-full bg-white transform transition-transform duration-200 shadow-sm ${instrumental ? 'translate-x-5' : 'translate-x-0'}`} />
                        </button>
                    </div>

                    <LyricsSection
                        showLyricsSub={showLyricsSub}
                        setShowLyricsSub={setShowLyricsSub}
                        instrumental={instrumental}
                        setInstrumental={setInstrumental}
                        lyrics={lyrics}
                        setLyrics={setLyrics}
                        lyricsRef={lyricsRef}
                        lyricsHeight={lyricsHeight}
                        startResizing={startResizing}
                        isFormattingLyrics={isFormattingLyrics}
                        handleFormat={handleFormat}
                        duration={duration > 0 ? duration : undefined}
                    />

                    <StyleSection
                        showStyleSub={showStyleSub}
                        setShowStyleSub={setShowStyleSub}
                        style={style}
                        setStyle={setStyle}
                        refreshMusicTags={refreshMusicTags}
                        isFormattingStyle={isFormattingStyle}
                        handleFormat={handleFormat}
                        styleRef={styleRef}
                        styleHeight={styleHeight}
                        startResizingStyle={startResizingStyle}
                        genreDropdownRef={genreDropdownRef}
                        showGenreDropdown={showGenreDropdown}
                        setShowGenreDropdown={setShowGenreDropdown}
                        selectedMainGenre={selectedMainGenre}
                        setSelectedMainGenre={setSelectedMainGenre}
                        selectedSubGenre={selectedSubGenre}
                        setSelectedSubGenre={setSelectedSubGenre}
                        getSubGenreCount={getSubGenreCount}
                        genreSearch={genreSearch}
                        setGenreSearch={setGenreSearch}
                        filteredCombinedGenres={filteredCombinedGenres}
                        subGenreDropdownRef={subGenreDropdownRef}
                        showSubGenreDropdown={showSubGenreDropdown}
                        setShowSubGenreDropdown={setShowSubGenreDropdown}
                        filteredSubGenres={filteredSubGenres}
                        musicTags={musicTags}
                        bpm={bpm}
                        keyScale={keyScale}
                        timeSignature={timeSignature}
                        effectiveBpm={effectiveBpm}
                        effectiveKeyScale={effectiveKeyScale}
                        triggerWord={triggerWord}
                    />

                    <MusicParametersSection
                        bpm={bpm}
                        setBpm={setBpm}
                        keyScale={keyScale}
                        setKeyScale={setKeyScale}
                        timeSignature={timeSignature}
                        setTimeSignature={setTimeSignature}
                        duration={duration}
                        setDuration={setDuration}
                        detectedBpm={detectedBpm}
                        detectedKey={detectedKey}
                        taskType={taskType}
                        sourceDuration={sourceDuration}
                        tempoScale={tempoScale}
                    />

                </div>
            )}
        </div>
    );
};

export default TrackDetailsAccordion;

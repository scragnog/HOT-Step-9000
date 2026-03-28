import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Eye, Loader2, Users, Sparkles, AlertTriangle } from 'lucide-react';
import { lireekApi, Profile } from '../../../services/lyricStudioApi';
import {
  useStreamingStore,
  startStreamBuildProfile,
} from '../../../stores/streamingStore';
import { StreamingPanel } from '../StreamingPanel';

interface ProfilesTabProps {
  lyricsSetId: number;
  profiles: Profile[];
  onRefresh: () => void;
  showToast: (msg: string) => void;
}

export const ProfilesTab: React.FC<ProfilesTabProps> = ({
  lyricsSetId, profiles, onRefresh, showToast,
}) => {
  const [building, setBuilding] = useState(false);
  const [selectedProfile, setSelectedProfile] = useState<Profile | null>(null);
  const [provider, setProvider] = useState('openrouter');
  const [model, setModel] = useState('');
  const streaming = useStreamingStore();

  const handleBuild = async () => {
    setBuilding(true);
    try {
      startStreamBuildProfile(lyricsSetId, { provider, model: model || undefined }, () => {
        // onComplete callback
        setBuilding(false);
        onRefresh();
        showToast('Profile built successfully');
      });
    } catch (err: any) {
      showToast(`Build failed: ${err.message}`);
      setBuilding(false);
    }
  };

  const handleDelete = async (profile: Profile) => {
    if (!confirm('Delete this profile?')) return;
    try {
      await lireekApi.deleteProfile(profile.id);
      showToast('Profile deleted');
      onRefresh();
    } catch (err: any) {
      showToast(`Failed: ${err.message}`);
    }
  };

  const formatProfileData = (data: Record<string, any>): string => {
    return JSON.stringify(data, null, 2);
  };

  return (
    <div className="p-4 space-y-4">
      {/* Build button */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleBuild}
          disabled={building}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-zinc-700 disabled:text-zinc-500 text-white text-sm font-semibold transition-all"
        >
          {building ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Building...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              Build New Profile
            </>
          )}
        </button>
      </div>

      {/* Streaming panel */}
      {streaming.visible && (
        <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 overflow-hidden">
          <StreamingPanel
            visible={streaming.visible}
            streamText={streaming.text}
            phase={streaming.phase}
            done={streaming.done}
          />
        </div>
      )}

      {/* Profile list */}
      {profiles.length === 0 && !building ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <div className="w-14 h-14 rounded-full bg-white/5 flex items-center justify-center mb-4">
            <Users className="w-7 h-7 text-zinc-600" />
          </div>
          <h3 className="text-base font-semibold text-zinc-400 mb-2">No profiles yet</h3>
          <p className="text-sm text-zinc-500 max-w-xs">
            Build a stylistic profile from the source lyrics to start generating.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {profiles.map((profile) => {
            const data = profile.profile_data;
            const themes = data?.themes as string[] | undefined;
            return (
              <div
                key={profile.id}
                className="group rounded-xl border border-white/5 hover:border-white/10 bg-white/[0.01] overflow-hidden transition-colors"
              >
                <div className="flex items-center gap-3 px-4 py-3">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs px-2 py-0.5 rounded-md bg-indigo-500/10 text-indigo-400 font-medium">
                        {profile.provider}
                      </span>
                      {profile.model && (
                        <span className="text-xs text-zinc-500 truncate">
                          {profile.model}
                        </span>
                      )}
                    </div>
                    {themes && themes.length > 0 && (
                      <p className="text-xs text-zinc-400 truncate">
                        {themes.slice(0, 4).join(', ')}
                      </p>
                    )}
                    <p className="text-[11px] text-zinc-600 mt-1">
                      {new Date(profile.created_at).toLocaleDateString()}
                    </p>
                  </div>
                  <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={() => setSelectedProfile(selectedProfile?.id === profile.id ? null : profile)}
                      className="p-2 rounded-lg hover:bg-white/5 text-zinc-400 hover:text-white transition-colors"
                      title="View profile"
                    >
                      <Eye className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(profile)}
                      className="p-2 rounded-lg hover:bg-red-500/10 text-zinc-400 hover:text-red-400 transition-colors"
                      title="Delete profile"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Expanded profile detail */}
                {selectedProfile?.id === profile.id && (
                  <div className="border-t border-white/5 p-4">
                    <pre className="text-xs text-zinc-300 whitespace-pre-wrap font-mono leading-relaxed max-h-80 overflow-y-auto bg-black/20 rounded-lg p-3">
                      {formatProfileData(profile.profile_data)}
                    </pre>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

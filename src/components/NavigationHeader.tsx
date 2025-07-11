
import { Moon, Sun, Download } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface NavigationHeaderProps {
  isDarkMode: boolean;
  toggleTheme: () => void;
}

const NavigationHeader = ({ isDarkMode, toggleTheme }: NavigationHeaderProps) => {
  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white/90 dark:bg-slate-900/90 backdrop-blur-sm border-b shadow-sm">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* Left: Title */}
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="font-semibold bg-blue-500 text-white border-blue-500">
              Live Dataset
            </Badge>
          </div>

          {/* Center: User Info */}
          <div className="flex flex-col items-center">
            <Badge className="mb-1 bg-slate-700 text-white">
              21MIC7040
            </Badge>
            <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
              Chandan Sathvik
            </span>
          </div>

          {/* Right: Controls */}
          <div className="flex items-center gap-3">
            {/* Theme Toggle */}
            <Button
              variant="outline"
              size="sm"
              onClick={toggleTheme}
              className="flex items-center gap-2"
            >
              {isDarkMode ? (
                <>
                  <Sun className="w-4 h-4" />
                  <span className="hidden sm:inline">Light</span>
                </>
              ) : (
                <>
                  <Moon className="w-4 h-4" />
                  <span className="hidden sm:inline">Dark</span>
                </>
              )}
            </Button>

            {/* Download Button */}
            <a href="/21mic7040_dataset.xlsx" download="21mic7040_dataset.xlsx">
  <Button size="sm" className="gap-2 bg-green-600 hover:bg-green-700 text-white">
    <Download className="w-4 h-4" />
    <span className="hidden sm:inline">Download Dataset</span>
  </Button>
</a>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default NavigationHeader;

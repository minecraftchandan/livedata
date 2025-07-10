
import { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useToast } from '@/hooks/use-toast';

interface CodeBlockProps {
  code: string;
  title?: string;
  language?: string;
}

const CodeBlock = ({ code, title, language = 'python' }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      toast({
        title: "Code copied!",
        description: "The code has been copied to your clipboard.",
      });
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast({
        title: "Failed to copy",
        description: "Could not copy code to clipboard.",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="relative border rounded-lg overflow-hidden bg-white dark:bg-slate-900 shadow-sm">
      {title && (
        <div className="flex items-center justify-between px-4 py-3 bg-slate-50 dark:bg-slate-800 border-b">
          <h4 className="font-medium text-sm text-slate-700 dark:text-slate-300">
            {title}
          </h4>
          <Button
            variant="ghost"
            size="sm"
            onClick={copyToClipboard}
            className="h-8 w-8 p-0"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4 text-slate-500" />
            )}
          </Button>
        </div>
      )}
      <div className="relative">
        <pre className="p-4 overflow-x-auto text-sm font-mono leading-relaxed bg-slate-50 dark:bg-slate-900">
          <code className={`language-${language} text-slate-800 dark:text-slate-200`}>{code}</code>
        </pre>
        {!title && (
          <Button
            variant="ghost"
            size="sm"
            onClick={copyToClipboard}
            className="absolute top-2 right-2 h-8 w-8 p-0 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm"
          >
            {copied ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4 text-slate-500" />
            )}
          </Button>
        )}
      </div>
    </div>
  );
};

export default CodeBlock;

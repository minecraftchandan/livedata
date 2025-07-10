
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface SectionBlockProps {
  title?: string;
  children: React.ReactNode;
  variant?: 'default' | 'primary' | 'accent';
  className?: string;
}

const SectionBlock = ({ title, children, variant = 'default', className }: SectionBlockProps) => {
  const variants = {
    default: 'border-slate-200 bg-white dark:border-slate-700 dark:bg-slate-800',
    primary: 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950/30',
    accent: 'border-purple-200 bg-purple-50 dark:border-purple-800 dark:bg-purple-950/30',
  };

  return (
    <Card className={cn(
      variants[variant], 
      'shadow-sm',
      className
    )}>
      {title && (
        <CardHeader className="pb-3">
          <CardTitle className="text-lg text-slate-800 dark:text-slate-200">
            {title}
          </CardTitle>
        </CardHeader>
      )}
      <CardContent className={title ? 'pt-3' : 'py-6'}>
        {children}
      </CardContent>
    </Card>
  );
};

export default SectionBlock;
